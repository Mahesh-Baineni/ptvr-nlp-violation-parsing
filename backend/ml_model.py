import json
import os
from typing import List, Tuple, Dict, Any
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    accuracy_score,
    f1_score,
    classification_report,
    precision_recall_curve,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import re

warnings.filterwarnings("ignore", module="sklearn")

# Role mappings
ROLE_NAME_TO_ID: Dict[str, int] = {
    "retailer": 1,
    "distributor": 2,
    "manufacturer": 3,
}
ROLE_ID_TO_NAME: Dict[int, str] = {v: k for k, v in ROLE_NAME_TO_ID.items()}


class ImprovedViolationMLModel:
    """
    Enhanced multi-label text classifier for violator roles.

    Key design choices (to reduce overfitting and leakage):
    - NO injection of role-derived keywords from `violatorTypeIDs`
    - NO use of violator names / city as training features
    - Uses narrative, product context, state, violationMaterialIDs, violationTypeID
    - Learns per-role thresholds from validation (no hard-coding)
    - Grouped train/val split by violator (company) to avoid overlap
    - Logistic Regression (balanced) or Gradient Boosting
    - Calibration analysis + plots

    threshold_beta:
        - beta = 1.0  → standard F1 (precision = recall)
        - beta < 1.0  → favors precision
        - beta > 1.0  → favors recall

    NOTE:
        - Per-role thresholds (self.role_thresholds) are learned in `train`
          and are intended for REVIEW / AUTO-APPROVAL decisions.
        - Label selection in `predict_roles` now uses a separate, looser
          BASE_LABEL_THRESHOLD so that some predicted roles can still fall
          below the review threshold and be flagged for human review.
    """

    def __init__(self, data_path: str, use_ensemble: bool = True, threshold_beta: float = 0.5):
        self.data_path = data_path
        self.use_ensemble = use_ensemble
        self.threshold_beta = threshold_beta

        # Word-level TF-IDF
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words="english",
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )
        # Char-level TF-IDF
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            max_features=5000,
            min_df=2,
            strip_accents="unicode",
            lowercase=True,
        )

        self.mlb = MultiLabelBinarizer()
        self.classifier: OneVsRestClassifier | None = None

        self.metrics: Dict[str, Any] = {}
        # Will be learned during training; initialize sane placeholders
        self.role_thresholds: Dict[str, float] = {name: 0.5 for name in ROLE_NAME_TO_ID}

    
    # Text preprocessing & dataset
    
    def preprocess_text(self, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s'\-]", " ", text)
        return text.strip()

    def compose_text_from_report(self, report: Dict[str, Any]) -> str:
        """
        Build a text representation of a report using ONLY information that is
        available at prediction time and does NOT directly leak the target label.

        Uses:
          - narrative / description
          - product type / subtype
          - tobacco brand
          - state name
          - violationMaterialIDs (as tokens)
          - violationTypeID (as tokens)

        """
        parts: List[str] = []
        violation = report.get("Violation") or {}

        # 1) Core narrative
        narrative = violation.get("description", "") or ""
        parts.append(self.preprocess_text(narrative))

        # 2) Product context
        product_type = violation.get("producttype", "") or ""
        product_subtype = violation.get("productsubtype", "") or ""
        tobacco_brand = violation.get("tobaccoBrand", "") or ""

        if product_type:
            parts.append(f"product {self.preprocess_text(product_type)}")
        if product_subtype:
            parts.append(f"subtype {self.preprocess_text(product_subtype)}")
        if tobacco_brand:
            parts.append(f"brand {self.preprocess_text(tobacco_brand)}")

        # 3) State
        st = violation.get("State") or report.get("State") or {}
        state_name = st.get("name", "") or ""
        if state_name and state_name.lower() != "none":
            parts.append(f"state {self.preprocess_text(state_name)}")

        # 4) Violation materials and types (structured → tokens)
        mat_ids = violation.get("violationMaterialIDs") or []
        if not isinstance(mat_ids, list):
            mat_ids = [mat_ids]
        for mid in mat_ids:
            if mid not in (None, ""):
                parts.append(f"material_{str(mid).strip()}")

        vt_ids = violation.get("violationTypeID") or []
        if not isinstance(vt_ids, list):
            vt_ids = [vt_ids]
        for vt in vt_ids:
            if vt not in (None, ""):
                parts.append(f"vtype_{str(vt).strip()}")

       

        return " ".join(str(p) for p in parts if p)

    def _load_dataset(self) -> Tuple[List[str], List[List[int]], List[str]]:
        """
        Load dataset and build:
          - texts: feature strings
          - labels: list of role ID lists
          - groups: group key per sample (for grouped train/val split)
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts: List[str] = []
        labels: List[List[int]] = []
        groups: List[str] = []

        reports = (
            data["reports"]
            if isinstance(data, dict) and "reports" in data
            else data
            if isinstance(data, list)
            else []
        )

        for rep_idx, rep in enumerate(reports):
            violation = rep.get("Violation", {}) or {}
            violators = violation.get("Violator", []) or []

            record_text = self.compose_text_from_report(rep)

            roles: List[int] = []
            violator_names: List[str] = []
            for v in violators:
                vt = v.get("violatorTypeIDs")
                if vt in (1, 2, 3):
                    roles.append(int(vt))
                name = (v.get("name") or "").strip().lower()
                if name:
                    violator_names.append(name)

            if record_text and roles:
                texts.append(record_text)
                labels.append(sorted(set(roles)))

                # Group key: all violator names for this report (to avoid overlap)
                if violator_names:
                    group_key = "|".join(sorted(set(violator_names)))
                else:
                    # Fallback to GUID or index if no names
                    group_key = str(rep.get("guid", f"idx_{rep_idx}"))
                groups.append(group_key)

        if not texts:
            raise ValueError("No training examples built from dataset.")
        print(f"Loaded {len(texts)} training examples")
        return texts, labels, groups

    # Model selection
    
    def create_classifier(self, X_train_shape: tuple) -> OneVsRestClassifier:
        n_samples, _ = X_train_shape
        if self.use_ensemble and n_samples >= 100:
            print("Using Gradient Boosting Classifier")
            base = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )
        else:
            print("Using Logistic Regression (balanced)")
            base = LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            )
        return OneVsRestClassifier(base, n_jobs=-1)

    
    # Training (with grouped split + learned thresholds)
    def train(self) -> None:
        print("IMPROVED VIOLATION ML MODEL TRAINING")
        texts, labels, groups = self._load_dataset()

        Y = self.mlb.fit_transform(labels)
        print("\nLabel distribution:")
        for i, role_id in enumerate(self.mlb.classes_):
            role_name = ROLE_ID_TO_NAME.get(role_id, str(role_id))
            count = int(Y[:, i].sum())
            pct = 100.0 * count / len(Y)
            print(f"  {role_name} (ID={role_id}): {count}/{len(Y)} ({pct:.1f}%)")

        # Grouped train/validation split by company/violator 
        try:
            unique_groups = np.array(sorted(set(groups)))
            train_groups, val_groups = train_test_split(
                unique_groups, test_size=0.3, random_state=42
            )
            train_idx = [i for i, g in enumerate(groups) if g in train_groups]
            val_idx = [i for i, g in enumerate(groups) if g in val_groups]

            X_train_texts = [texts[i] for i in train_idx]
            X_val_texts = [texts[i] for i in val_idx]
            Y_train = Y[train_idx]
            Y_val = Y[val_idx]

            print(
                "\nUsing GROUPED split by violator/company "
                f"({len(train_groups)} train groups, {len(val_groups)} val groups)"
            )
        except Exception as e:
            print(f"[WARN] Grouped split failed ({e}); falling back to random split.")
            try:
                X_train_texts, X_val_texts, Y_train, Y_val = train_test_split(
                    texts, Y, test_size=0.3, random_state=42, stratify=Y
                )
            except ValueError:
                print("Stratification not possible; using random split.")
                X_train_texts, X_val_texts, Y_train, Y_val = train_test_split(
                    texts, Y, test_size=0.3, random_state=42
                )

        print(f"\nTrain size: {len(X_train_texts)} | Val size: {len(X_val_texts)}")

        # Features
        print("FEATURE EXTRACTION")
        X_train_word = self.word_vectorizer.fit_transform(X_train_texts)
        X_val_word = self.word_vectorizer.transform(X_val_texts)
        X_train_char = self.char_vectorizer.fit_transform(X_train_texts)
        X_val_char = self.char_vectorizer.transform(X_val_texts)

        from scipy.sparse import hstack

        X_train = hstack([X_train_word, X_train_char])
        X_val = hstack([X_val_word, X_val_char])
        print(f"Combined features: {X_train.shape}")

        # Fit
        print("MODEL TRAINING")
        self.classifier = self.create_classifier(X_train.shape)
        self.classifier.fit(X_train, Y_train)
        print("Training complete!")

        # Evaluate
        print("MODEL EVALUATION")
        Y_pred = self.classifier.predict(X_val)
        Y_prob = self.classifier.predict_proba(X_val)

        role_names = [ROLE_ID_TO_NAME[i] for i in self.mlb.classes_]
        print("\nPer-role Classification Report:")
        print(
            classification_report(
                Y_val, Y_pred, target_names=role_names, zero_division=0
            )
        )

        acc = accuracy_score(Y_val, Y_pred)
        micro_f1 = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(Y_val, Y_pred, average="macro", zero_division=0)

        print("\nOVERALL METRICS")
        print(f"Accuracy: {acc:.3f}")
        print(f"Micro F1: {micro_f1:.3f}")
        print(f"Macro F1: {macro_f1:.3f}")

        # Label-combination report
        true_ids = self.mlb.inverse_transform(Y_val)
        pred_ids = self.mlb.inverse_transform(Y_pred)

        def ids_to_names(id_list_list):
            out = []
            for ids in id_list_list:
                out.append("none" if not ids else ",".join(ROLE_ID_TO_NAME[i] for i in ids))
            return out

        true_names = ids_to_names(true_ids)
        pred_names = ids_to_names(pred_ids)
        combo_report = classification_report(
            true_names, pred_names, output_dict=True, zero_division=0
        )

        self.metrics = {
            "accuracy": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "classification_report": combo_report,
        }

        print("\nFull Classification Report (label combinations):")
        print(classification_report(true_names, pred_names, zero_division=0))

        # ------------------------------------------------------------------
        # Calibration + learned thresholds
        # ------------------------------------------------------------------
        print("\nCALIBRATION & THRESHOLDS")
        plots_dir = os.path.join(
            os.path.dirname(self.data_path), "calibration_plots_improved"
        )
        os.makedirs(plots_dir, exist_ok=True)

        calibration_info: Dict[str, Dict[str, float]] = {}
        learned_thresholds: Dict[str, float] = {}

        for j, role_id in enumerate(self.mlb.classes_):
            role = ROLE_ID_TO_NAME.get(role_id, str(role_id))
            y_true = Y_val[:, j]
            y_prob = Y_prob[:, j]

            if y_true.sum() == 0:
                print(f"[SKIP] '{role}' has no positives in validation.")
                continue

            # Brier & calibration curve
            brier = brier_score_loss(y_true, y_prob)
            try:
                n_bins = max(5, min(10, int(max(5, y_true.sum()))))
                frac_pos, mean_pred = calibration_curve(
                    y_true, y_prob, n_bins=n_bins, strategy="quantile"
                )

                plt.figure(figsize=(8, 6))
                plt.plot(mean_pred, frac_pos, "o-", label="Model", linewidth=2)
                plt.plot([0, 1], [0, 1], "--", label="Perfect", linewidth=2)
                plt.title(f"Calibration – {role.title()}")
                plt.xlabel("Predicted Probability")
                plt.ylabel("Observed Fraction Positive")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plot_path = os.path.join(plots_dir, f"calibration_{role}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()

                calibration_info[role] = {
                    "brier_score": float(brier),
                    "mean_confidence": float(np.mean(y_prob)),
                    "mean_positive_confidence": float(
                        np.mean(y_prob[y_true == 1])
                    ),
                    "plot": plot_path,
                }
                print(f"[{role}] Brier={brier:.3f} | Saved {plot_path}")
            except Exception as e:
                print(f"[WARN] Calibration plotting failed for {role}: {e}")

            # Learn threshold: maximize F_beta per role (data-driven)
            best_thr = 0.5
            best_score = -1.0
            try:
                precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
                beta = self.threshold_beta
                beta2 = beta * beta

                # thresholds has length len(precisions) - 1
                for p, r, t in zip(precisions[1:], recalls[1:], thresholds):
                    if p == 0.0 and r == 0.0:
                        continue
                    # F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
                    num = (1.0 + beta2) * p * r
                    den = (beta2 * p + r) + 1e-8
                    f_beta = num / den
                    if f_beta > best_score:
                        best_score = f_beta
                        best_thr = float(t)

                # Fallback if we never improved best_score
                if best_score < 0.0:
                    pos_probs = y_prob[y_true == 1]
                    best_thr = float(np.median(pos_probs)) if len(pos_probs) else 0.5
            except Exception as e:
                print(f"[WARN] Threshold sweep failed for {role}: {e}")
                pos_probs = y_prob[y_true == 1]
                best_thr = float(np.median(pos_probs)) if len(pos_probs) else 0.5
                best_score = -1.0

            best_thr = float(min(max(best_thr, 0.05), 0.95))  # clamp
            learned_thresholds[role] = best_thr
            print(
                f"[{role}] Learned threshold={best_thr:.3f} "
                f"(best F_beta={best_score:.3f}, beta={self.threshold_beta})"
            )

        if learned_thresholds:
            self.role_thresholds.update(learned_thresholds)

        print("\nSUGGESTED PER-ROLE THRESHOLDS")
        for role, t in self.role_thresholds.items():
            print(f"  {role}: {t:.3f}")

        thresholds_path = os.path.join(plots_dir, "optimal_thresholds.json")
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(self.role_thresholds, f, indent=2)
        print(f"Saved thresholds → {thresholds_path}")

        self.metrics["calibration"] = calibration_info
        self.metrics["thresholds"] = self.role_thresholds

    # Inference
    def predict_roles(self, text: str) -> Tuple[List[str], Dict[str, float], float]:
        """
        Predict violator roles for a free-text description.

        - Uses a BASE_LABEL_THRESHOLD to decide which roles are *predicted*.
        - Learned per-role thresholds (self.role_thresholds) are NOT used here;
          they are applied later in the pipeline to decide auto-approval vs
          human review.
        """
        if not hasattr(self.classifier, "predict_proba"):
            raise RuntimeError("Model not trained. Call .train() first.")

        X_word = self.word_vectorizer.transform([text])
        X_char = self.char_vectorizer.transform([text])
        from scipy.sparse import hstack

        X = hstack([X_word, X_char])

        prob_vec = self.classifier.predict_proba(X)[0]

        # Base threshold used ONLY for label selection
        BASE_LABEL_THRESHOLD = 0.30

        preds_binary = np.zeros(len(prob_vec), dtype=int)
        for i, (role_id, prob) in enumerate(zip(self.mlb.classes_, prob_vec)):
            if prob >= BASE_LABEL_THRESHOLD:
                preds_binary[i] = 1

        # Ensure at least one positive
        if preds_binary.sum() == 0 and len(prob_vec):
            preds_binary[int(np.argmax(prob_vec))] = 1

        predicted_ids = [
            self.mlb.classes_[i] for i, v in enumerate(preds_binary) if v == 1
        ]
        predicted_roles = [ROLE_ID_TO_NAME.get(i, str(i)) for i in predicted_ids]

        probs_by_role: Dict[str, float] = {}
        for i, p in enumerate(prob_vec):
            role_id = self.mlb.classes_[i]
            role_name = ROLE_ID_TO_NAME.get(role_id, str(role_id))
            probs_by_role[role_name] = float(p)

        max_conf = float(np.max(prob_vec)) if len(prob_vec) else 0.0
        return predicted_roles, probs_by_role, max_conf

    def predict_roles_for_report(
        self, report: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, float], float]:
        text = self.compose_text_from_report(report)
        return self.predict_roles(text)
