# SGD, MOMENTUM & ADAM | EXAMPLE A
# Gradient descent on a quadratic
# Dr. Nagendrababu Vanamala

theta = 5.0
for step in range(30):
    gradient = 2 * (theta - 2)
    theta -= 0.1 * gradient
print("Parameter:", round(theta, 6))
print("Loss:", round((theta - 2) ** 2, 8))
