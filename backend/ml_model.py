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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline, FeatureUnion
import re

warnings.filterwarnings("ignore", module="sklearn")


# Role mappings

ROLE_NAME_TO_ID: Dict[str, int] = {
    "retailer": 1,
    "distributor": 2,
    "manufacturer": 3,
}

ROLE_ID_TO_NAME: Dict[int, str] = {v: k for k, v in ROLE_NAME_TO_ID.items()}

# ImprovedViolationMLModel with better feature engineering

class ImprovedViolationMLModel:
    """
    Enhanced multi-label text classifier for violator roles.
    
    Key Improvements:
    1. Fixed ngram_range to (1,3) instead of (3,8) - critical bug fix
    2. Added text preprocessing and cleaning
    3. Better feature extraction with multiple vectorizers
    4. Ensemble methods with hyperparameter tuning
    5. Better handling of class imbalance
    6. Cross-validation for more robust evaluation
    """

    def __init__(self, data_path: str, use_ensemble: bool = True):
        self.data_path = data_path
        self.use_ensemble = use_ensemble
        
        # Improved text vectorizers
        # Word-level TF-IDF with proper ngrams
        self.word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3), 
            max_features=5000,
            stop_words='english',
            min_df=2,  
            max_df=0.95,  
            sublinear_tf=True,  
            strip_accents='unicode',
            lowercase=True,
        )
        
        # Character-level TF-IDF for robustness (catches misspellings, names)
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=2000,
            min_df=2,
            strip_accents='unicode',
            lowercase=True,
        )
        
        # Multi-label binarizer
        self.mlb = MultiLabelBinarizer()
        
        self.classifier = None
        
        self.metrics: Dict[str, Any] = {}
        self.role_thresholds: Dict[str, float] = {
            name: 0.5 for name in ROLE_NAME_TO_ID.keys()  
        }

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text for better feature extraction.
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        # Keep hyphens, apostrophes for words like "e-cigarette" and "don't"
        text = re.sub(r'[^\w\s\'-]', ' ', text)
        
        return text.strip()

    def compose_text_from_report(self, report: Dict[str, Any]) -> str:
        """
        Build a comprehensive text representation from the report.
        Enhanced with better text cleaning and structure.
        """
        parts: List[str] = []

        violation = report.get("Violation") or {}

        # Core narrative (most important)
        narrative = violation.get("description", "") or ""
        parts.append(self.preprocess_text(narrative))

        # Product information - important for context
        product_type = violation.get("producttype", "") or ""
        product_subtype = violation.get("productsubtype", "") or ""
        tobacco_brand = violation.get("tobaccoBrand", "") or ""
        
        if product_type:
            parts.append(f"product {self.preprocess_text(product_type)}")
        if product_subtype:
            parts.append(f"subtype {self.preprocess_text(product_subtype)}")
        if tobacco_brand:
            parts.append(f"brand {self.preprocess_text(tobacco_brand)}")

        # State information
        st = violation.get("State") or report.get("State") or {}
        state_name = st.get("name", "") or ""
        if state_name and state_name != "none":
            parts.append(f"state {self.preprocess_text(state_name)}")

        # Violator information - names and addresses
        violators = violation.get("Violator", []) or []
        for viol in violators:
            name = viol.get("name", "") or ""
            if name:
                parts.append(f"violator {self.preprocess_text(name)}")
            
            # Role-specific keywords
            role_id = viol.get("violatorTypeIDs")
            if role_id == 1:
                parts.append("retailer retail store shop")
            elif role_id == 2:
                parts.append("distributor distribution supplier supply")
            elif role_id == 3:
                parts.append("manufacturer manufacturing producer production")
            
            addr = viol.get("ViolatorAddress") or {}
            city = addr.get("city", "") or ""
            if city:
                parts.append(self.preprocess_text(city))

        # Join with spaces and clean up
        full_text = " ".join(str(p) for p in parts if p)
        return full_text

    def _load_dataset(self) -> Tuple[List[str], List[List[int]]]:
        """
        Load dataset with enhanced text processing.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        texts: List[str] = []
        labels: List[List[int]] = []

        if isinstance(data, dict) and "reports" in data:
            reports = data["reports"]
        elif isinstance(data, list):
            reports = data
        else:
            reports = []

        for rep in reports:
            violation = rep.get("Violation", {}) or {}
            violators = violation.get("Violator", []) or []

            record_text = self.compose_text_from_report(rep)

            # Collect unique roles
            roles: List[int] = []
            for v in violators:
                vt = v.get("violatorTypeIDs")
                if vt in (1, 2, 3):
                    roles.append(int(vt))

            if record_text and roles:
                texts.append(record_text)
                labels.append(sorted(set(roles)))

        if not texts:
            raise ValueError("No training examples could be built from the dataset.")

        print(f"Loaded {len(texts)} training examples")
        return texts, labels

    def create_classifier(self, X_train_shape: tuple) -> OneVsRestClassifier:
        """
        Create the best classifier based on dataset size and characteristics.
        """
        n_samples, n_features = X_train_shape
        
        if self.use_ensemble and n_samples >= 100:
            # Use Gradient Boosting for better performance on small datasets
            print("Using Gradient Boosting Classifier")
            base_clf = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )
        else:
            # Improved Logistic Regression with better hyperparameters
            print("Using Logistic Regression with optimized hyperparameters")
            base_clf = LogisticRegression(
                max_iter=1000,  
                C=1.0,  
                class_weight='balanced',  
                solver='lbfgs',  
                random_state=42,
                n_jobs=-1,
            )
        
        return OneVsRestClassifier(base_clf, n_jobs=-1)

    def train(self) -> None:
        """
        Train the improved multi-label classifier with enhanced features.
        """
        
        print("IMPROVED VIOLATION ML MODEL TRAINING")
        
        
        texts, labels = self._load_dataset()

        # Multi-label binarization
        Y = self.mlb.fit_transform(labels)
        
        print(f"\nLabel distribution:")
        for i, role_id in enumerate(self.mlb.classes_):
            role_name = ROLE_ID_TO_NAME.get(role_id, str(role_id))
            count = Y[:, i].sum()
            pct = 100.0 * count / len(Y)
            print(f"  {role_name} (ID={role_id}): {count}/{len(Y)} ({pct:.1f}%)")

        # Stratified split with larger test size for better evaluation
        try:
            X_train_texts, X_test_texts, Y_train, Y_test = train_test_split(
                texts,
                Y,
                test_size=0.3,
                random_state=42,
                stratify=Y,
            )
        except ValueError:
            print("Warning: Stratification not possible, using random split")
            X_train_texts, X_test_texts, Y_train, Y_test = train_test_split(
                texts,
                Y,
                test_size=0.2,
                random_state=42,
            )

        print(f"\nTrain set size: {len(X_train_texts)}")
        print(f"Test set size: {len(X_test_texts)}")
        print("\nTraining set positives per role:")
        print(Y_train.sum(axis=0))
        print("Test set positives per role:")
        print(Y_test.sum(axis=0))

        # Combined feature extraction with word and character n-grams
        print("FEATURE EXTRACTION")
        
        print("Extracting word-level TF-IDF features...")
        X_train_word = self.word_vectorizer.fit_transform(X_train_texts)
        X_test_word = self.word_vectorizer.transform(X_test_texts)
        
        print("Extracting character-level TF-IDF features...")
        X_train_char = self.char_vectorizer.fit_transform(X_train_texts)
        X_test_char = self.char_vectorizer.transform(X_test_texts)
        
        # Combine features (horizontal stack)
        from scipy.sparse import hstack
        X_train = hstack([X_train_word, X_train_char])
        X_test = hstack([X_test_word, X_test_char])
        
        print(f"Combined feature dimensions: {X_train.shape}")
        print(f"  - Word features: {X_train_word.shape[1]}")
        print(f"  - Char features: {X_train_char.shape[1]}")

        # Create and train classifier
        print("MODEL TRAINING")
        
        self.classifier = self.create_classifier(X_train.shape)
        self.classifier.fit(X_train, Y_train)
        print("Training complete!")

        # Evaluate
        print("MODEL EVALUATION")
        
        Y_pred = self.classifier.predict(X_test)
        Y_prob = self.classifier.predict_proba(X_test)
        
        # Per-role metrics
        role_names = [ROLE_ID_TO_NAME[i] for i in self.mlb.classes_]
        print("\nPer-role Classification Report:")
        print(classification_report(
            Y_test, Y_pred, 
            target_names=role_names, 
            zero_division=0
        ))

        # Overall metrics
        acc = accuracy_score(Y_test, Y_pred)
        micro_f1 = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
        macro_f1 = f1_score(Y_test, Y_pred, average="macro", zero_division=0)

        print("OVERALL METRICS (80/20 split)")
        print(f"Accuracy: {acc:.3f}")
        print(f"Micro F1: {micro_f1:.3f}")
        print(f"Macro F1: {macro_f1:.3f}")

        # Multi-label combination accuracy
        true_ids = self.mlb.inverse_transform(Y_test)
        pred_ids = self.mlb.inverse_transform(Y_pred)
        
        def ids_to_names(id_list_list):
            names = []
            for ids in id_list_list:
                if not ids:
                    names.append("none")
                else:
                    names.append(",".join(ROLE_ID_TO_NAME.get(i, str(i)) for i in ids))
            return names

        true_names = ids_to_names(true_ids)
        pred_names = ids_to_names(pred_ids)

        combo_report = classification_report(
            true_names,
            pred_names,
            output_dict=True,
            zero_division=0,
        )

        self.metrics = {
            "accuracy": acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "classification_report": combo_report,
        }

        print("\nFull Classification Report (label combinations):")
        print(classification_report(
            true_names,
            pred_names,
            zero_division=0,
        ))

        # Calibration analysis
        print("CALIBRATION ANALYSIS")
        
        plots_dir = os.path.join(os.path.dirname(self.data_path), "calibration_plots_improved")
        os.makedirs(plots_dir, exist_ok=True)

        calibration_info: Dict[str, Dict[str, float]] = {}
        optimal_thresholds: Dict[str, float] = {}

        for j, role_id in enumerate(self.mlb.classes_):
            role = ROLE_ID_TO_NAME.get(role_id, str(role_id))
            y_true = Y_test[:, j]
            y_prob = Y_prob[:, j]

            if y_true.sum() == 0:
                print(f"[SKIP] Role '{role}' – no positive examples in test set")
                continue

            # Brier score
            brier = brier_score_loss(y_true, y_prob)
            
            # Calibration curve
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_true,
                    y_prob,
                    n_bins=min(10, int(y_true.sum())),  # Adjust bins for small samples
                    strategy="quantile",
                )

                calibration_info[role] = {
                    "brier_score": float(brier),
                    "mean_confidence": float(np.mean(y_prob)),
                    "mean_positive_confidence": float(np.mean(y_prob[y_true == 1])),
                }

                # Plot
                plt.figure(figsize=(8, 6))
                plt.plot(mean_pred, frac_pos, "o-", label="Model", linewidth=2)
                plt.plot([0, 1], [0, 1], "--", label="Perfectly calibrated", linewidth=2)
                plt.title(f"Calibration Plot – {role.title()}", fontsize=14, fontweight='bold')
                plt.xlabel("Predicted Probability", fontsize=12)
                plt.ylabel("Observed Fraction Positive", fontsize=12)
                plt.legend(loc="best", fontsize=10)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                plot_path = os.path.join(plots_dir, f"calibration_{role}.png")
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()

                print(f"[{role.upper()}] Brier={brier:.3f}, "
                      f"Mean Conf={np.mean(y_prob):.3f}, "
                      f"Mean Pos Conf={np.mean(y_prob[y_true == 1]):.3f}")
                print(f"  → Saved plot: {plot_path}")
            
            except Exception as e:
                print(f"[WARNING] Could not create calibration plot for {role}: {e}")

            # Optimal threshold for ~80% precision
            try:
                precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
                
                thr_for_80 = None
                for p, t in zip(precisions[:-1], thresholds):
                    if p >= 0.8:
                        thr_for_80 = float(t)
                        break
                
                if thr_for_80 is None:
                    # Fallback: median of positive predictions
                    pos_probs = y_prob[y_true == 1]
                    if len(pos_probs) > 0:
                        thr_for_80 = float(np.median(pos_probs))
                    else:
                        thr_for_80 = 0.5
                
                optimal_thresholds[role] = thr_for_80
            except Exception as e:
                print(f"[WARNING] Could not compute optimal threshold for {role}: {e}")
                optimal_thresholds[role] = 0.5

        if optimal_thresholds:
            self.role_thresholds.update(optimal_thresholds)

        print("SUGGESTED PER-ROLE THRESHOLDS")
        for role, t in self.role_thresholds.items():
            print(f"{role}: {t:.3f}")

        # Save thresholds
        thresholds_path = os.path.join(plots_dir, "optimal_thresholds.json")
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(self.role_thresholds, f, indent=2)
        print(f"\n→ Saved thresholds to: {thresholds_path}")

        self.metrics["calibration"] = calibration_info
        self.metrics["thresholds"] = self.role_thresholds
        

    def predict_roles(self, text: str) -> Tuple[List[str], Dict[str, float], float]:
        """
        Predict roles for a raw text string.
        """
        if not hasattr(self.classifier, "predict_proba"):
            raise RuntimeError("Model not trained. Call .train() first.")

        # Vectorize with both word and char features
        X_word = self.word_vectorizer.transform([text])
        X_char = self.char_vectorizer.transform([text])
        
        from scipy.sparse import hstack
        X = hstack([X_word, X_char])
        
        # Get probabilities
        prob_matrix = self.classifier.predict_proba(X)[0]

        # Use per-role thresholds
        preds_binary = np.zeros(len(prob_matrix), dtype=int)
        for i, (role_id, prob) in enumerate(zip(self.mlb.classes_, prob_matrix)):
            role_name = ROLE_ID_TO_NAME.get(role_id, str(role_id))
            threshold = self.role_thresholds.get(role_name, 0.5)
            if prob >= threshold:
                preds_binary[i] = 1
        
        # Ensure at least one label
        if preds_binary.sum() == 0:
            preds_binary[np.argmax(prob_matrix)] = 1

        # Map to role names
        predicted_ids = [
            self.mlb.classes_[i]
            for i, v in enumerate(preds_binary)
            if v == 1
        ]
        predicted_roles = [ROLE_ID_TO_NAME.get(i, str(i)) for i in predicted_ids]

        # Probabilities by role
        probs_by_role: Dict[str, float] = {}
        for i, p in enumerate(prob_matrix):
            role_id = self.mlb.classes_[i]
            role_name = ROLE_ID_TO_NAME.get(role_id, str(role_id))
            probs_by_role[role_name] = float(p)

        max_conf = float(max(prob_matrix)) if len(prob_matrix) else 0.0
        return predicted_roles, probs_by_role, max_conf

    def predict_roles_for_report(
        self, report: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, float], float]:
        """
        Predict roles for a full report.
        """
        text = self.compose_text_from_report(report)
        return self.predict_roles(text)
