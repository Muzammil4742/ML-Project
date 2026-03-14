
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
from PIL import Image, ImageDraw


mnist = fetch_openml('mnist_784', as_frame=False)
x, y = mnist['data'], mnist['target'].astype(int)


some_digit = x[3600].reshape(28,28)
plt.imshow(some_digit, cmap='binary')
plt.axis("off")
plt.show()
print("Label:", y[3600])


shuffle_index = np.random.permutation(60000)
x_train, x_test = x[:60000][shuffle_index], x[60000:]
y_train, y_test = y[:60000][shuffle_index], y[60000:]


clf = OneVsRestClassifier(
    LogisticRegression(tol=0.1, max_iter=1000, solver='saga')
)
clf.fit(x_train, y_train)


pred_train = clf.predict(x_train[:1000])  
print("Sample train accuracy:", accuracy_score(y_train[:1000], pred_train))


cv_scores = cross_val_score(clf, x_train[:5000], y_train[:5000], cv=3, scoring='accuracy')
print("Cross-validation accuracy:", cv_scores.mean())


joblib.dump(clf, "mnist_logistic_model.pkl")
print("Model saved as mnist_logistic_model.pkl")


clf_loaded = joblib.load("mnist_logistic_model.pkl")


pred = clf_loaded.predict([x_test[0]])
plt.imshow(x_test[0].reshape(28,28), cmap='binary')
plt.axis("off")
plt.show()
print("Predicted label:", pred[0], "Actual label:", y_test[0])


def draw_digit_predict():
    
    img = Image.new('L', (28,28), color=0)
    draw = ImageDraw.Draw(img)
    print("Draw your digit (0-9) by filling white pixels on black background.")

   
    try:
        user_img = Image.open("my_digit.png").convert('L').resize((28,28))
        user_arr = np.array(user_img).reshape(1,784)
        user_arr = 255 - user_arr  # invert if necessary
        user_arr = user_arr / 255.0 * 255  # scale like MNIST
        pred_user = clf_loaded.predict(user_arr)
        plt.imshow(user_img, cmap='binary')
        plt.axis("off")
        plt.show()
        print("Predicted digit:", pred_user[0])
    except FileNotFoundError:
        print("Please create a 28x28 black & white image named 'my_digit.png' in the folder.")


draw_digit_predict()
