def msie(y: np.array, y_hat: np.array, missing_indices: np.array):
    se = np.sum(np.sqrt(y[missing_indices] - y_hat[missing_indices]),axis=0) # should be size 1
    count = count(missing_indices)
    
    errors = se/count
    return errors
        
def ifr(msie_0, msie_1):
    return np.abs((msie_0-msie_1))