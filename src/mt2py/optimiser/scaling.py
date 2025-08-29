import numpy as np


def scale_linear(p,lower,upper):
    return (p-lower)/(upper-lower)

def unscale_linear(p,lower,upper):
    return p*(upper-lower) + lower

def d_scale_linear(p,lower,upper):
    return upper- lower

def scale_log(p,lower,upper):
    return(np.log(p)-np.log(lower))/(np.log(upper)-np.log(lower))

def unscale_log(p,lower,upper):
    return np.power(upper/lower,p)*lower

def d_scale_log(p,lower,upper):
    return lower*np.power(upper/lower,p)*np.log(upper/lower)

def scale_exp(p,lower,upper):
    return(np.exp(p)-np.exp(lower))/(np.exp(upper)-np.exp(lower))

def unscale_exp(p,lower,upper):
    return np.log(p*(np.exp(upper)-np.exp(lower))+np.exp(lower))

def d_scale_exp(p,lower,upper):
    return (np.exp(upper)-np.exp(lower))/(p*(np.exp(upper)-np.exp(lower))+np.exp(lower))

def unscale_params(p,lower,upper,scaling):

    unscaled_params = np.empty_like(p)
    unscaled_derivatives = np.empty_like(p)
    for i,param in enumerate(p):
        if scaling[i] == 'lin':
            p_scale = unscale_linear(param,lower[i],upper[i])
            unscaled_params[i] = p_scale
            unscaled_derivatives[i] = d_scale_linear(p_scale,lower[i],upper[i])
        elif scaling[i] =='log':
            p_scale = unscale_log(param,lower[i],upper[i])
            unscaled_params[i] = p_scale
            unscaled_derivatives[i] = d_scale_log(p_scale,lower[i],upper[i])
        elif scaling[i] =='exp':
            p_scale = unscale_exp(param,lower[i],upper[i])
            unscaled_params[i] = p_scale
            unscaled_derivatives[i] = d_scale_exp(p_scale,lower[i],upper[i])
        else:
            print('Unrecognised Scaling, should be lin, log or exp')
    return unscaled_params,unscaled_derivatives

def scale_params(p,lower,upper,scaling):

    scaled_params = np.empty_like(p)
    scaled_derivatives = np.empty_like(p)
    for i,param in enumerate(p):
        if scaling[i] == 'lin':
            p_scale = scale_linear(param,lower[i],upper[i])
            scaled_params[i] = p_scale
            scaled_derivatives[i] = d_scale_linear(p_scale,lower[i],upper[i])
        elif scaling[i] =='log':
            p_scale = scale_log(param,lower[i],upper[i])
            scaled_params[i] = p_scale
            scaled_derivatives[i] = d_scale_log(p_scale,lower[i],upper[i])
        elif scaling[i] =='exp':
            p_scale = scale_exp(param,lower[i],upper[i])
            scaled_params[i] = p_scale
            scaled_derivatives[i] = d_scale_exp(p_scale,lower[i],upper[i])
        else:
            print('Unrecognised Scaling, should be lin, log or exp')
    return scaled_params,scaled_derivatives