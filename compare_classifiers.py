"""import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('diabetes.csv')
print(df.head())
print(df.describe())
print(df.shape)
print(df.isnull().sum())

columns_to_check = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI" ]
for col in columns_to_check:
    zero_counts = (df[col] == 0).sum()
    zero_percantage = 100 * zero_counts / len(df)
    print(f"{col}: {zero_counts} % {zero_percantage:.2f}")

# "SkinThickness" ve "Insulin" sütunlarını drop edelim

df.drop("SkinThickness", axis=1, inplace=True)
df.drop("Insulin", axis=1, inplace=True)
print(df.shape)

from sklearn.model_selection import train_test_split

# 1. Veriyi Ayırma
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

# Doldurulacak sütunlar
columns_to_check2 = ["Glucose", "BloodPressure", "BMI"]

# --- FOR DÖNGÜSÜ İLE MANUEL IMPUTATION ---

for col in columns_to_check2:
    # ADIM A: Önce her iki setteki 0'ları NaN (Boş) yapalım
    # (Bunu yapmazsak medyan hesabına 0'lar dahil olur, sonuç yanlış çıkar)
    X_train[col] = X_train[col].replace(0, np.nan)
    X_test[col] = X_test[col].replace(0, np.nan)

    # ADIM B: Medyanı SADECE TRAIN setinden hesapla
    # (Test seti henüz yokmuş gibi davranıyoruz)
    median_val = X_train[col].median()

    # ADIM C: Train setindeki boşlukları bu değerle doldur
    X_train[col] = X_train[col].fillna(median_val)

    # ADIM D: Test setindeki boşlukları da TRAIN'den bulduğumuz değerle doldur
    # (Test setinin kendi medyanını KULLANMIYORUZ!)
    X_test[col] = X_test[col].fillna(median_val)

    print(f"{col} için Train Medyanı: {median_val:.1f} -> Train ve Test'e uygulandı.")

# --- KONTROL ---
print("\nİşlem Sonrası Train Setindeki 0 Sayısı:")
print((X_train[columns_to_check2] == 0).sum())

print("\nİşlem Sonrası Test Setindeki 0 Sayısı:")
print((X_test[columns_to_check2] == 0).sum())

print(X_train.describe())
print(X_test.describe())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def calculate_model_metrics(true, predicted):
    accuracy = accuracy_score(true, predicted)
    confusion = confusion_matrix(true, predicted)
    classification = classification_report(true, predicted)
    return accuracy, confusion, classification

models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "SVC": SVC(probability=True), # probability=True ROC-AUC için gerekli
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()
}

# Sonuçları toplu görmek için listemizi tutalım
results = []

print(f"{'Model':<20} {'Accuracy':<10}")
print("=" * 60)

for name, model in models.items():

    # 1. Veri Seçimi (Scaled vs Original)
    if name in ["Decision Tree", "Random Forest", "AdaBoost"]:
        current_X_train, current_X_test = X_train, X_test
    else:
        current_X_train, current_X_test = X_train_scaled, X_test_scaled

    # 2. Eğitim
    model.fit(current_X_train, y_train)

    # 3. Tahmin
    y_pred = model.predict(current_X_test)
    acc = accuracy_score(y_test, y_pred)

    results.append({"Model": name, "Accuracy": acc})

    # --- RAPORLAMA KISMI ---
    print(f"\n🔵 MODEL: {name}")
    print(f"Accuracy: {acc:.4f}")
    print("-" * 30)

    # Classification Report (Recall ve Precision değerleri burada)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix (Hatalar nerede yapıldı?)
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print("=" * 60)  # Modeller arası ayraç

print("-----Hyperparameter tuning sonrası-----")

param_grids = {
    "Logistic Regression": {
        'penalty' : ['l1', 'l2', 'elasticnet'],
        'c_values' :  [100, 10, 1, 0.1, 0.01],
        'solver' : ['newton-cg', 'liblinear', 'sag', 'saga', 'lbfgs', 'newton-cholesky']
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance']
    },
    "SVC": {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "Naive Bayes": {}, # Naive Bayes genelde parametre istemez (varsayılan iyidir)
    "Decision Tree": {
        "criterion" : ["gini", "entropy", "log_loss"],
        "splitter" : ["best", "random"],
        "max_depth" : [1,2,3,4,5,15,None],
        "max_features" : ["sqrt", "log2", None]
    },
    "Random Forest": {
        "n_estimators" : [100,200,300,500,1000],
        "max_depth" : [5,8,19,15, None],
        "max_features" : ["log2", "sqrt", 5,6,7,8],
        "min_samples_split" : [2,8,15,20]
    },
    "AdaBoost": {
        "n_estimators": [50,70,100,120,150,200],
        "learning_rate": [0.001,0.01,0.1,1,10,100]
    }
}

results_grid = []

print(f"{'Model':<20} {'Accuracy':<10} {'Best Params'}")
print("=" * 100)

for name, model in models.items():

    # A) Veri Seçimi (Scaled vs Original)
    if name in ["Decision Tree", "Random Forest", "AdaBoost"]:
        current_X_train, current_X_test = X_train, X_test
    else:
        current_X_train, current_X_test = X_train_scaled, X_test_scaled

    # B) Hyperparameter Tuning (GridSearchCV)
    # Eğer parametre listesi boş değilse GridSearch yap, boşsa (NB gibi) düz eğit
    param_grid = param_grids.get(name, {})

    if param_grid:  # Parametre varsa Tuning yap
        grid = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid.fit(current_X_train, y_train)

        best_model = grid.best_estimator_  # En iyi ayarları bulmuş model
        best_params = grid.best_params_
    else:  # Parametre yoksa düz eğit
        best_model = model
        best_model.fit(current_X_train, y_train)
        best_params = "Default"

    # C) Tahmin ve Raporlama
    y_pred = best_model.predict(current_X_test)
    acc = accuracy_score(y_test, y_pred)

    results_grid.append({"Model": name, "Accuracy": acc, "Best Params": str(best_params)})

    # Ekrana Özet Bas
    print(f"{name:<20} {acc:.4f}     {str(best_params)[:60]}...")

    # Detaylı Rapor (İstersen burayı yorum satırından çıkar)
    print(f"\nConfusion Matrix ({name}):")
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)

# --- 4. SONUÇLARI GÖRSELLEŞTİRME ---

results_df = pd.DataFrame(results_grid).sort_values(by="Accuracy", ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x="Accuracy", y="Model", data=results_df, palette="magma")
plt.title("Tuning Sonrası Model Karşılaştırması")
plt.xlim(0.65, 0.90)
plt.show()

# En iyi parametreleri tam liste olarak görelim
print("\n KAZANANLARIN TAM PARAMETRELERİ:")
for index, row in results_df.iterrows():
    print(f"▶ {row['Model']} (Acc: {row['Accuracy']:.4f}):")
    print(f"  {row['Best Params']}\n")

"""
"""
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
penalty = ['l1', 'l2', 'elasticnet']
c_values = [100, 10, 1, 0.1, 0.01]
solver = ['newton-cg', 'liblinear', 'sag', 'saga', 'lbfgs', 'newton-cholesky']

params = dict(penalty=penalty, C=c_values, solver=solver)
print(params)

# grid search cv
from sklearn.model_selection import GridSearchCV, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True) #bunlar default değerler zaten

grid = GridSearchCV(estimator=logistic, param_grid=params, cv=cv, scoring="accuracy", n_jobs=-1) # cv yapmak istemezsek yapmayabiliriz, eğer istiyorsak da StratifiedKFold ile yapabiliriz
#n_jobs=-1 cpuda ne kadar yer varsa kullan demek

grid.fit(X_train, y_train)

print(grid.best_params_)
print(grid.best_score_)

y_pred_logistic = grid.predict(X_test)
acc_score_logistic = accuracy_score(y_test, y_pred_logistic)
print("Accuracy score: ",acc_score_logistic)
print(classification_report(y_test, y_pred_logistic))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_logistic))

print("------Hyperparameter tuning sonrası logistic------")

logistic = LogisticRegression(C= 0.1, penalty='l2', solver='newton-cg', random_state=15)
logistic.fit(X_train, y_train)
y_pred_logistic = logistic.predict(X_test)
acc_score_logistic = accuracy_score(y_test, y_pred_logistic)
print("Accuracy score: ",acc_score_logistic)
print(classification_report(y_test, y_pred_logistic))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_logistic))

print("------SVM Classifier------")

from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
acc_score_svc = accuracy_score(y_test, y_pred_svc)
print("Accuracy score: ",acc_score_svc)
print(classification_report(y_test, y_pred_svc))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_svc))

param_grid = {
    'kernel': ['rbf'],
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ["scale", "auto"]
}

grid_svc = GridSearchCV(svc, param_grid = param_grid, cv=5)
grid_svc.fit(X_train, y_train)
print(grid_svc.best_params_)
print(grid_svc.best_score_)

print("------Hyperparameter tuning sonrası SVC------")

svc = SVC(C= 1, kernel = 'rbf', gamma = 'auto', random_state=15)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
acc_score_svc = accuracy_score(y_test, y_pred_svc)
print("Accuracy score: ",acc_score_svc)
print(classification_report(y_test, y_pred_svc))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_svc))

print("------Naive Bayes Classifier------")

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_score_nb = accuracy_score(y_test, y_pred_nb)
print("Accuracy score: ",acc_score_nb)
print(classification_report(y_test, y_pred_nb))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_nb))

print("------KNN Classifier------")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_score_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy score: ",acc_score_knn)
print(classification_report(y_test, y_pred_knn))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_knn))

print("------Decision Tree Classifier------")

from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_score_tree = accuracy_score(y_test, y_pred_tree)
print("Accuracy score: ",acc_score_tree)
print(classification_report(y_test, y_pred_tree))


params = {
    "criterion" : ["gini", "entropy", "log_loss"],
    "splitter" : ["best", "random"],
    "max_depth" : [1,2,3,4,5,15,None],
    "max_features" : ["sqrt", "log2", None]
}
grid_tree = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=5, n_jobs=-1, scoring="accuracy")
grid_tree.fit(X_train, y_train)
print(grid_tree.best_params_)
print(grid_tree.best_score_)

y_pred_grid_tree = grid_tree.predict(X_test)
acc_score_grid_tree = accuracy_score(y_test, y_pred_grid_tree)
print("Accuracy score: ",acc_score_grid_tree)
print(classification_report(y_test, y_pred_grid_tree))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_grid_tree))

print("------Hyperparameter tuning sonrası Decision Tree Classifier------")

tree = DecisionTreeClassifier(criterion= 'gini', max_depth= 5, max_features= None, splitter= 'best', random_state= 15)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
acc_score_tree = accuracy_score(y_test, y_pred_tree)
print("Accuracy score: ",acc_score_tree)
print(classification_report(y_test, y_pred_tree))

print("------Random Forest Classifier------")

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred_rf = rfc.predict(X_test)
acc_score_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy score: ",acc_score_rf)
print(classification_report(y_test, y_pred_rf))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_rf))

rf_params ={
    "n_estimators" : [100,200,300,500,1000],
    "max_depth" : [5,8,19,15, None],
    "max_features" : ["log2", "sqrt", 5,6,7,8],
    "min_samples_split" : [2,8,15,20]
}

grid_rf = GridSearchCV(RandomForestClassifier(), param_grid=rf_params, cv=5, n_jobs=-1, scoring="accuracy")
grid_rf.fit(X_train, y_train)
print(grid_rf.best_params_)
print(grid_rf.best_score_)
y_pred_grid_rf = grid_rf.predict(X_test)
acc_score_grid_rf = accuracy_score(y_test, y_pred_grid_rf)
print("Accuracy score: ",acc_score_grid_rf)
print(classification_report(y_test, y_pred_grid_rf))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_grid_rf))

print("------Hyperparameter tuning sonrası Random Forest Classifier------")

rfc = RandomForestClassifier(max_depth= 19, max_features= 7, min_samples_split= 2, n_estimators= 100, random_state= 15)
rfc.fit(X_train, y_train)
y_pred_rf = rfc.predict(X_test)
acc_score_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy score: ",acc_score_rf)
print(classification_report(y_test, y_pred_rf))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_rf))

print("------AdaBoost Classifier------")

from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier()
adaboost.fit(X_train, y_train)
y_pred_ada = adaboost.predict(X_test)
acc_score_ada = accuracy_score(y_test, y_pred_ada)
print("Accuracy score: ",acc_score_ada)
print(classification_report(y_test, y_pred_ada))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_ada))

ada_params = {
    "n_estimators": [50,70,100,120,150,200],
    "learning_rate": [0.001,0.01,0.1,1,10,100]
}

grid_ada = GridSearchCV(estimator = AdaBoostClassifier(), param_grid = ada_params, verbose  = 1, cv = 5, n_jobs = -1)
grid_ada.fit(X_train, y_train)
print(grid_ada.best_params_)
print(grid_ada.best_score_)
y_pred_grid_ada = grid_ada.predict(X_test)
acc_score_grid_ada = accuracy_score(y_test, y_pred_grid_ada)
print("Accuracy score: ",acc_score_grid_ada)
print(classification_report(y_test, y_pred_grid_ada))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_grid_ada))

print("------Hyperparameter tuning sonrası AdaBoost Classifier------")

adaboost = AdaBoostClassifier(learning_rate= 0.1, n_estimators= 100)
adaboost.fit(X_train, y_train)
y_pred_ada = adaboost.predict(X_test)
acc_score_ada = accuracy_score(y_test, y_pred_ada)
print("Accuracy score: ",acc_score_ada)
print(classification_report(y_test, y_pred_ada))
print("Confusion_matrix: \n",confusion_matrix(y_test, y_pred_ada))"""