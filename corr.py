# Simple python function to compute Pearson correlation
def corr(x,y):
  nx = len(x)
  ny = len(y)
  if nx != ny:
      return 0
  if nx == 0:
      return 0
  N = float(nx)
  # compute mean of each vector
  meanx = sum(x) / N
  meany = sum(y) / N
# compute standard deviation of each vector
sdx = math.sqrt(sum([(a-meanx)*(a-meanx) for a in x])/(N-1) ) sdy = math.sqrt(sum([(a-meany)*(a-meany) for a in y])/(N-1) )
# normalise vector elements to zero mean and unit variance normx = [(a-meanx)/sdx for a in x]
normy = [(a-meany)/sdy for a in y]
# return the Pearson correlation coefficient
return sum([normx[i]*normy[i] for i in range(nx)])/(N-1)