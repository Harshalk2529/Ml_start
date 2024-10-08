import cv2
import pandas as pd
import joblib

model = joblib.load('color_classifier_knn.pkl')

img_path = r"C:\Users\Hi\Desktop\Color-Detection-OpenCV-main\Screenshot 2024-05-10 091150.png"
img = cv2.imread(img_path)

clicked = False
r = g = b = x_pos = y_pos = 0

index = ["color", "color_name", "hex", "R", "G", "B"]
# csv = pd.read_csv(r'C:\Users\Hi\Desktop\Color-Detection-OpenCV-main\colors.csv', names=index, header=None)

def get_color_name(R, G, B):
    # Use the model to predict the color
    color_prediction = model.predict([[R, G, B]])
    return color_prediction[0]

def draw_function(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        global b, g, r, x_pos, y_pos, clicked
        clicked = True
        x_pos = x
        y_pos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_function)

while True:
    cv2.imshow("image", img)
    if clicked:
        # Draw a rectangle with the clicked color
        cv2.rectangle(img, (20, 20), (750, 60), (b, g, r), -1)

        # Predict color name using the machine learning model
        color_name = get_color_name(r, g, b)

        # Display the color name and RGB values
        text = color_name + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
        cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if r + g + b >= 600:
            cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        clicked = False

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
