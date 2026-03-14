import tkinter as tk
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from lenet_models import ModifiedLeNet5

# Canvas dimensions
WIDTH, HEIGHT = 280, 280

class DigitDrawer:
    def __init__(self, root):
        self.root = root
        self.root.title("LeNet-5 Digit Recognizer")

        # Main frames
        self.leftFrame = tk.Frame(root, padx=10, pady=10)
        self.leftFrame.grid(row=0, column=0)
        self.rightFrame = tk.Frame(root, padx=10, pady=10)
        self.rightFrame.grid(row=0, column=1, sticky="n")

        # Prediction label
        self.resultLabel = tk.Label(self.leftFrame, text="Draw a digit", font=("Arial", 20, "bold"))
        self.resultLabel.pack(pady=5)

        # Feature Maps Label
        self.featureMapsLabel= tk.Label(self.rightFrame, text="Prediction & Feature Mapping", font =("Arial",20,"bold"))
        self.featureMapsLabel.pack(pady=(10,0))

        # Canvas for drawing
        self.canvas = tk.Canvas(self.leftFrame, width=WIDTH, height=HEIGHT, bg="#ffffff", bd=3, relief="ridge")
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.onMouseRelease)

        # Clear button
        self.clearButton = tk.Button(self.leftFrame, text="Clear", font=("Arial", 12), command=self.clearCanvas)
        self.clearButton.pack(pady=10)

        # PIL image for drawing
        self.image = Image.new("L", (WIDTH, HEIGHT), color=240)
        self.drawImage = ImageDraw.Draw(self.image)

        # Load model
        self.loadModel()

        # Embedded probability chart
        #self.ProbabilityLabel = tk.Label(self.rightFrame,text="Probabilities", font=("Arial",14,"bold"))
        self.prob_fig, self.prob_ax = plt.subplots(figsize=(4,2))
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, master=self.rightFrame)
        self.prob_canvas.get_tk_widget().pack(pady=10)

        # Feature maps title
        self.featureMapsLabel = tk.Label(self.rightFrame, text="Convolutional Layer One Feature Maps", font=("Arial", 14, "bold"))
        self.featureMapsLabel.pack(pady=(10, 0))  # small padding above the plot

        # Embedded Conv1 feature maps
        self.conv1_fig, self.conv1_axs = plt.subplots(2, 3, figsize=(6,4))
        self.conv1_canvas = FigureCanvasTkAgg(self.conv1_fig, master=self.rightFrame)
        self.conv1_canvas.get_tk_widget().pack(pady=10)

    # Load pre-trained model
    def loadModel(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ModifiedLeNet5().to(self.device)
        self.model.load_state_dict(torch.load("saved_models/modified_lenet5.pth", map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    # Draw on canvas
    def draw(self, event):
        r = 4
        x, y = event.x, event.y
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black", outline="black")
        self.drawImage.ellipse([x-r, y-r, x+r, y+r], fill=0)

    # Mouse release triggers prediction
    def onMouseRelease(self, event):
        self.updatePrediction()

    # Clear canvas
    def clearCanvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (WIDTH, HEIGHT), color=240)
        self.drawImage = ImageDraw.Draw(self.image)
        self.resultLabel.config(text="Draw a digit")
        self.prob_ax.cla()
        self.prob_canvas.draw()
        for ax in self.conv1_axs.flatten():
            ax.cla()
        self.conv1_canvas.draw()

    # Preprocess image
    def preprocessImage(self):
        img = np.array(self.image)
        img = 255 - img
        blank, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]
        img = cv2.resize(img, (20,20))
        canvas = np.zeros((32,32), dtype=np.uint8)
        x_offset = (32-20)//2
        y_offset = (32-20)//2
        canvas[y_offset:y_offset+20, x_offset:x_offset+20] = img
        img = self.transform(canvas).unsqueeze(0)
        return img.to(self.device)

    # Get Conv1 feature maps
    def getFeatureMaps(self, img):
        with torch.no_grad():
            conv1 = torch.tanh(self.model.conv1(img))
            conv2 = torch.tanh(self.model.conv2(self.model.pool1(conv1)))
        return conv1, conv2

    # Update predictions and visualizations
    def updatePrediction(self):
        img = self.preprocessImage()
        with torch.no_grad():
            output = self.model(img)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]
            pred = probs.argmax()
        self.resultLabel.config(text=f"Prediction: {pred}")
        self.showProbabilities(probs)
        conv1, blank = self.getFeatureMaps(img)
        self.showFeatureMaps(conv1)

    # Show probability chart
    def showProbabilities(self, probs):
        self.prob_ax.cla()
        self.prob_ax.bar(range(10), probs, color="skyblue")
        self.prob_ax.set_ylim(0,1)
        self.prob_ax.set_xlabel("Digit")
        self.prob_ax.set_ylabel("Probability")
        self.prob_ax.set_title("Prediction Probabilities")
        self.prob_canvas.draw()

    # Show first 6 Conv1 feature maps
    def showFeatureMaps(self, conv1):
        #self.featureMapsLabel.config(text = "Title")
        conv_maps = conv1.squeeze(0).cpu()[:6]
        for i, ax in enumerate(self.conv1_axs.flatten()):
            ax.cla()
            if i < conv_maps.shape[0]:
                ax.imshow(conv_maps[i].detach().numpy(), cmap="viridis")
            ax.axis("off")
        self.conv1_canvas.draw()

def main():
    root = tk.Tk()
    app = DigitDrawer(root)
    root.mainloop()

if __name__ == "__main__":
    main()