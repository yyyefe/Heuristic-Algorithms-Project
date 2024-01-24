import woa_GradientBoosting
import GWO_GradientBoosting
import DE_GradientBoosting
import matplotlib.pyplot as plt


GWO, best_hyperparameters, best_accuracy=GWO_GradientBoosting.baslat()
print("GWO")
print("Best Parameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)

WOA, best_hyperparameters, best_accuracy=woa_GradientBoosting.baslat()
print("WOA")
print("Best Parameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)

DE, best_hyperparameters, best_accuracy=DE_GradientBoosting.baslat()
print("DE")
print("Best Parameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)

# İki farklı boyutta çizgisel grafik oluştur
plt.plot(WOA, label='WOA', color='blue', linestyle='-', marker='o')  
plt.plot(DE, label='DE', color='red', linestyle='-', marker='o')
plt.plot(GWO, label='GWO', color='green', linestyle='-', marker='o')

# Grafik başlığı ve etiketler
plt.title('Gradient Boosting')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')

# Gösterge
plt.legend()

# Grafik gösterimi
plt.show()