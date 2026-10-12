# SGD, MOMENTUM & ADAM | EXAMPLE B
# Momentum on the same quadratic
# Dr. Nagendrababu Vanamala

theta, velocity = 5.0, 0.0
for step in range(100):
    gradient = 2 * (theta - 2)
    velocity = 0.9 * velocity + 0.1 * gradient
    theta -= 0.1 * velocity
print("Momentum parameter:", round(theta, 6))
