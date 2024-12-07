import numpy as np

def msie(y: np.array, y_hat: np.array, missing_indices: np.array):
    squared_errors = np.square(y[missing_indices].values - y_hat[missing_indices].values)
    msie_value = np.mean(squared_errors)
    
    return msie_value
        
def ifr(msie_0, msie_1):
    return np.abs((msie_0-msie_1))

def calculate_msie(original_data, missing_data, imputed_data, features, sex=None):
    def get_masks(missing_data, features):
        missing_masks = [missing_data[feature]=='null' for feature in features]
        combined_missing_mask = np.logical_or.reduce(missing_masks)

        return combined_missing_mask

    def get_group_data(original_data, imputed_data, sex):
        if sex==None:
            group_indices = original_data['sex'].str.lower() != None
            group_original = original_data[group_indices]
            group_imputed = imputed_data[group_indices]
        
        else:
            group_indices = original_data['sex'].str.lower()  == sex
            group_original = original_data[group_indices]
            group_imputed = imputed_data[group_indices]
        
        return group_indices, group_original, group_imputed
    
    missing_mask = get_masks(missing_data, features)
    
    indices, original, imputed = get_group_data(original_data, imputed_data, sex)
    missing_mask = (missing_mask) & (indices)
    
    msie_result = msie(original[features], imputed[features], missing_mask)
    
    return msie_result

def calculate_ifr(original_data, missing_data, imputed_data, features):
    msie_male = calculate_msie(original_data, missing_data, imputed_data, features, 'male')
    msie_female = calculate_msie(original_data, missing_data, imputed_data, features, 'female')
    
    result = ifr(msie_male, msie_female)
    
    return result