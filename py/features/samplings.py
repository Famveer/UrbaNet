# oversampling and undersampling
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

class FeatureSampling:
    
    def SmoteSampling(self, X, y, random_state=42):
        sm=SMOTE(random_state=random_state, k_neighbors=len(np.unique(y)))
        X_over, y_over = sm.fit_resample(X, y)
        return (X_over, y_over)

    def overSampling(self, X, y, random_state=42):
        oversample = RandomOverSampler(random_state=random_state)
        X_over, y_over = oversample.fit_resample(X, y)
        return (X_over, y_over)

    def underSampling(self, X, y, random_state=42):
        undersample = RandomUnderSampler(random_state=random_state)
        X_under, y_under = undersample.fit_resample(X, y)
        return (X_under, y_under)

