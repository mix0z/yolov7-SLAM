class Yolov7MaskHandler():
    def __init__(self, model_path, device='cpu'):
        self.model = torch.jit.load(model_path, map_location=device)
        self.device = device
        self.model.to(device)
        self.model.eval()
        self.input_size = 640
        self.input_mean = [0.485, 0.456, 0.406]
        self.input_std = [0.229, 0.224, 0.225]

    def preprocess(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = np.float32(image) / 255
        image -= self.input_mean
        image /= self.input_std
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        image = torch.from_numpy(image)
        image = image.to(self.device)
        return image

    def postprocess(self, outputs):
        outputs = outputs.cpu().detach().numpy()
        outputs = np.squeeze(outputs)
        outputs = np.transpose(outputs, (1, 2, 0))
        outputs = cv2.resize(outputs, (640, 640))
        outputs = np.argmax(outputs, axis=-1)
        return outputs

    def inference(self, image):
        image = self.preprocess(image)
        outputs = self.model(image)
        outputs = self.postprocess(outputs)
        return outputs