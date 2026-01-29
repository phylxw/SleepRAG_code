import math




def _get_bemr_moments(alpha, beta):

    total = alpha + beta
    
    mean = alpha / total
    
    
    
    variance = (alpha * beta) / ((total ** 2) * (total + 1))
    
    return mean, variance

def _calculate_bemr_final_score(alpha, beta, cfg):

    mean, variance = _get_bemr_moments(alpha, beta)
    
    
    lambda1 = cfg.parameters.get('bemr_lambda1', 1.0)
    lambda2 = cfg.parameters.get('bemr_lambda2', 0.5) 
    
    
    
    
    final_score = (lambda1 * mean) + (lambda2 * math.sqrt(variance))
    
    return final_score