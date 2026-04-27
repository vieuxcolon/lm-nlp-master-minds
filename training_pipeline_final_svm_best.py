#nlp\training_pipeline_final_svm_best.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

# =========================================================
# LOAD RAW DATASET
# =========================================================
df = pd.read_csv("stack-overflow-data.csv")
df.columns = ["text", "tag"]

# Remove accidental header rows
df = df[df["tag"] != "tags"]

# Normalize labels
df["tag"] = df["tag"].str.lower()

# =========================================================
# MERGE DOTNET FAMILY (CRITICAL FIX)
# =========================================================
df["tag"] = df["tag"].replace({
    "c#": "dotnet_family",
    ".net": "dotnet_family",
    "asp.net": "dotnet_family"
})

# =========================================================
# SIMPLE CLEANING (NO TOKEN NOISE)
# =========================================================
def clean(text):
    if pd.isna(text):
        return ""
    return text.lower()

df["text"] = df["text"].apply(clean)

# =========================================================
# LEVEL 1: DOMAIN LABELING
# =========================================================
def level1_map(tag):
    frontend = {"html", "css", "javascript", "jquery", "angularjs"}
    mobile = {"ios", "iphone", "objective-c", "android"}

    if tag in frontend:
        return "frontend"
    elif tag in mobile:
        return "mobile"
    else:
        return "backend"

df["level1"] = df["tag"].apply(level1_map)

# =========================================================
# TRAIN LEVEL 1 MODEL
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["level1"],
    test_size=0.25,
    random_state=42,
    stratify=df["level1"]
)

level1_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        stop_words="english"
    )),
    ("clf", LinearSVC())
])

level1_model.fit(X_train, y_train)

print("\n=== LEVEL 1 EVALUATION ===")
preds = level1_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# =========================================================
# LEVEL 2: BACKEND MODEL
# =========================================================
backend_df = df[df["level1"] == "backend"]

X_train, X_test, y_train, y_test = train_test_split(
    backend_df["text"],
    backend_df["tag"],
    test_size=0.25,
    random_state=42,
    stratify=backend_df["tag"]
)

backend_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(C=1.2))
])

backend_model.fit(X_train, y_train)

print("\n=== BACKEND MODEL ===")
preds = backend_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# =========================================================
# LEVEL 2: FRONTEND MODEL
# =========================================================
frontend_df = df[df["level1"] == "frontend"]

X_train, X_test, y_train, y_test = train_test_split(
    frontend_df["text"],
    frontend_df["tag"],
    test_size=0.25,
    random_state=42,
    stratify=frontend_df["tag"]
)

frontend_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LinearSVC())
])

frontend_model.fit(X_train, y_train)

print("\n=== FRONTEND MODEL ===")
preds = frontend_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# =========================================================
# LEVEL 2: MOBILE MODEL (OPTIMAL STRUCTURE FIX)
# =========================================================
mobile_df = df[df["level1"] == "mobile"]

# MERGE IOS FAMILY INTO APPLE (KEY IMPROVEMENT)
mobile_df["tag"] = mobile_df["tag"].replace({
    "ios": "apple",
    "iphone": "apple",
    "objective-c": "apple"
})

X_train, X_test, y_train, y_test = train_test_split(
    mobile_df["text"],
    mobile_df["tag"],
    test_size=0.25,
    random_state=42,
    stratify=mobile_df["tag"]
)

mobile_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )),
    ("clf", LinearSVC())
])

mobile_model.fit(X_train, y_train)

print("\n=== MOBILE MODEL ===")
preds = mobile_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

# =========================================================
# FINAL PREDICTION FUNCTION
# =========================================================
def predict(text):
    text = text.lower()

    domain = level1_model.predict([text])[0]

    if domain == "backend":
        return backend_model.predict([text])[0]

    elif domain == "frontend":
        return frontend_model.predict([text])[0]

    else:
        return mobile_model.predict([text])[0]

# =========================================================
# SAMPLE PREDICTIONS
# =========================================================
tests = [
    "how to use entity framework in mvc c#",
    "css flexbox center div issue",
    "android intent not working in activity",
    "swift ios view controller crash",
    "python pandas groupby dataframe"
]

print("\n=== SAMPLE PREDICTIONS ===")
for t in tests:
    print(t, "→", predict(t))

# =========================================================
# FINAL TAG GROUP STRUCTURE REPORT
# =========================================================

LABEL_GROUPS = {
    "backend": sorted(backend_df["tag"].unique()),
    "frontend": sorted(frontend_df["tag"].unique()),
    "mobile": sorted(mobile_df["tag"].unique())
}

print("\n==============================")
print("FINAL TAG GROUP STRUCTURE")
print("==============================")

for group, tags in LABEL_GROUPS.items():
    print(f"\n{group.upper()} ({len(tags)} tags):")
    print("--------------------------------------------------")
    print(", ".join(tags))
