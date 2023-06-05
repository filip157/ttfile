import matplotlib.pyplot as plt


x =[70, 80, 90, 100, 110, 120, 130, 140, 150]
y = [22, 5.3, 11.5, 19.7, 22.9, 19.6, 11.2, 5.5, 2.1]
plt.figure(figsize = (8,4))
plt.xlabel("Smarts")
plt.ylabel("Probability (%)")
plt.title("Bar of IQ")
plt.show()