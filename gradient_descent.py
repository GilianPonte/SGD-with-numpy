def gradient_descent(X, y, num_it, lr, batch_size, momentum):
  import numpy as np
  np.random.seed(1994)
  # initial weights
  theta_hat = np.random.normal(0,1,2)

  # lists to store learning process
  log, theta_0, theta_1, mse = [], [], [], []
  b_w = 0
  for i in range(0,num_it):
    # batch_size
    idx = np.random.randint(0, len(X), batch_size)
    Xs = X[idx]
    ys = y[idx]
    n = len(X)

    # predict y
    y_hat = Xs.dot(theta_hat)

    # gradient of e'e w.r.t. theta_hat = (y - y_hat)^2 = gradient of y'y-2*theta_hat'X'y + theta_hat'X'Xtheta_hat w.r.t theta_hat =
    gradient_w = -2 * Xs.T.dot(ys) + 2*Xs.T.dot(y_hat)
    b_w = (momentum * b_w) + gradient_w

    # weight update
    theta_hat = theta_hat - (lr * b_w)

    log.append(theta_hat)
    theta_0.append(theta_hat[0])
    theta_1.append(theta_hat[1])
    loss = (y_hat - ys).sum()**2
    mse.append(loss)
    print(i," MSE =", np.round((loss).sum()**2,2), "and theta =", np.round_(theta_hat,1), "and gradient =", np.round(b_w,0))
  return theta_hat, log, mse, theta_0, theta_1
