## Function that defines Lens model parameters

def lens_model__(theta_E,gamma,e1,e2):
  kwargs_epl = {'theta_E': theta_E, 'gamma': gamma, 'center_x': 0, 'center_y': 0, 'e1': e1, 'e2': e2} 
  return kwargs_epl

## Function that defines Source models parameters

def source_model__(source_type,amp,e1,e2,center_x,center_y):

  if source_type == 'SERSIC_ELLIPSE':
    kwargs_sersic = {'amp': amp, 'R_sersic': 0.1, 'n_sersic': 1.5, 'e1':e1,'e2':e2,'center_x': center_x, 'center_y': center_y}
    return kwargs_sersic 

  if source_type == 'HERNQUIST':
    kwargs_Hernquist={ 'amp': amp,'Rs':0.71,'center_x': center_x, 'center_y': center_y}
    return kwargs_Hernquist
  
  if source_type=='POWER_LAW':
    kwargs_power_law={'amp': amp, 'gamma':0.1, 'e1':e1,'e2':e2, 'center_x': center_x, 'center_y': center_y}
    return kwargs_power_law

  if source_type=='ELLIPSOID':
    kwargs_ellipsoid={'amp': amp, 'radius':0.1, 'e1':e1,'e2':e2, 'center_x': center_x, 'center_y': center_y}
    return kwargs_ellipsoid

  if source_type=='UNIFORM':
    kwargs_UNIFORM={'amp': amp}
    return kwargs_UNIFORM

  if source_type=='GAUSSIAN_ELLIPSE':
    kwargs_gaussian_ellipse={'amp': amp, 'sigma':0.11, 'e1':e1,'e2':e2, 'center_x': center_x, 'center_y': center_y}
    return kwargs_gaussian_ellipse

  if source_type=='CHAMELEON':
    w_c=0.1
    w_t=1
    kwargs_chameleon={'amp':amp, 'w_c':w_c, 'w_t':w_t, 'e1':e1, 'e2':e2, 'center_x':center_x, 'center_y': center_y}
    return kwargs_chameleon
    
  if source_type=='MULTI_GAUSSIAN_ELLIPSE':
    kwargs_gaussian_multi_ellipse={'amp': [amp,0.1*amp,0.5*amp], 'sigma':[0.11,0.3,0.8], 'e1':e1,'e2':e2, 'center_x': center_x, 'center_y': center_y}
    return kwargs_gaussian_multi_ellipse
  
  if source_type=='NIE':
    s_scale=1
    kwargs_NIE={'amp': amp, 'e1':e1,'e2':e2, 's_scale':0.5, 'center_x': center_x, 'center_y': center_y}
    return kwargs_NIE
  
  if source_type=='PJAFFE_ELLIPSE':
    Ra=1
    Rs=5
    kwargs_jaffe={'amp': amp,'Ra':Ra, 'Rs':Rs, 'e1':e1,'e2':e2, 'center_x': center_x, 'center_y': center_y}
    return kwargs_jaffe
