# 피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여 데이터셋 재구성후 각모델별로 돌려서 결과 도출
# 기존 모델결과와 결과비교

#2. 모델구성   -> feature importance는 트리계열에만 있음
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# 결과비교
# 1.DecisionTree
# 기존 acc
# 컬럼삭제 후 acc
