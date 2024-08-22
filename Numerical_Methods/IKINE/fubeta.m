function hb=fubeta(x,beta)

hb=0.5*(1+tanh(1/(1-x/beta)-beta/x));