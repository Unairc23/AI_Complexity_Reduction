import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import random, numpy as np

import wandb
# Para instalar torch junto con cuda:
# "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126"

# Check if the current `accelerator <https://pytorch.org/docs/stable/torch.html#accelerators>`__
# is available, and if not, use the CPU
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")

torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Below we are preprocessing data for CIFAR-10. We use an arbitrary batch size of 128.
transforms_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loading the CIFAR-10 dataset:
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms_cifar)

#Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

# ========================== MODELOS ============================
# Deeper neural network class to be used as teacher:
class DeepNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN, self).__init__()
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 2
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 3
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Bloque 4 (opcional, si tu GPU lo permite)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # Clasificador completamente conectado
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DeepNN_Adaptada(nn.Module):
    def __init__(self, num_classes=10):
        super(DeepNN_Adaptada, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # En teoria en todos estos ReLU puedes usar inplace=True, lo que hace que los valores nuevos sobrescriban
            # los anteriores, lo cual reduce la memoria necesaria. Esto puede crear problemas, pero al ser una red
            # simple como esta no deberia de haber problemas
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Se adapta el tamaño a 7x7 (explicar)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))  # out: 64 x 7 x 7

        # Clasificador totalmente conectado: fc3136 -> fc1200 -> fc800
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 3136),
            nn.ReLU(inplace=True),

            nn.Linear(3136, 1200),
            nn.ReLU(inplace=True),

            nn.Linear(1200, 800),
            nn.ReLU(inplace=True),

            nn.Linear(800, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Lightweight neural network class to be used as student:
class LightNN(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class LightNN_Adaptada(nn.Module):
    def __init__(self, num_classes=10):
        super(LightNN_Adaptada, self).__init__()
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8

            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8 -> 4x4
        )

        # Clasificador totalmente conectado
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),  # 2048 -> 256
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        """
        patience: nº de épocas sin mejorar antes de parar
        min_delta: mejora mínima requerida
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def step(self, loss, model):
        if loss + self.min_delta < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

# ========================== ENTRENAMIENTO Y EVALUACION ============================
def train(model, train_loader, epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # inputs: A collection of batch_size images
            # labels: A vector of dimensionality batch_size with integers denoting class of each image
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: Output of the network for the collection of images. A tensor of dimensionality batch_size x num_classes
            # labels: The actual labels of the images. Vector of dimensionality batch_size
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader)}")

def test(model, test_loader, device):
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# ========================== ENTRENAMIENTO KD ============================
def train_knowledge_distillation(teacher, student, train_loader, epochs, learning_rate, T, alpha, device):
    ce_loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    early_stopper = EarlyStopping(patience=3)

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            # Forward pass with the student model
            student_logits = student(inputs)

            # Loss KD estándar usando KLDiv entre log_softmax(student/T) y softmax(teacher/T).
            log_p_student = F.log_softmax(student_logits / T, dim=1)
            q_teacher = F.softmax(teacher_logits / T, dim=1)

            kd_loss = F.kl_div(log_p_student, q_teacher, reduction='batchmean') * (T * T)
            label_loss = ce_loss(student_logits, labels)

            loss = alpha * kd_loss + (1.0 - alpha) * label_loss

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}")

        if early_stopper.step(epoch_loss, student):
            print("Early stopping triggered!")
            break

    # Restaurar mejores pesos
    early_stopper.restore(student)

def train_kd_wandb(teacher, student):
    wandb.init()
    config = wandb.config

    train_knowledge_distillation(
        teacher=teacher,
        student=student,
        train_loader=train_loader,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        T=config.T,
        alpha=config.alpha,
        device=device
    )

    # Evaluación del modelo estudiante
    acc = test(student, test_loader, device)

    # Reportar métrica al sweep
    wandb.log({"accuracy": acc})

def load_model(model_class, path, device):
    """Carga un modelo si el archivo existe, o devuelve uno nuevo."""
    model = model_class().to(device)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f" Modelo cargado desde {path}")
    else:
        print(f" No se encontró el archivo {path}, se inicializa un modelo nuevo.")
    return model

if __name__ == "__main__":
    # Creo que esta linea solo es encesaria en windows, linux hace un fork
    torch.multiprocessing.freeze_support()

    print(sys.argv)

    if len(sys.argv)>1 and sys.argv[1] == "full":
        torch.manual_seed(42)
        teacher = DeepNN_Adaptada(num_classes=10).to(device)
        train(teacher, train_loader, epochs=50, learning_rate=0.01, device=device)

        torch.manual_seed(42)
        student = LightNN_Adaptada(num_classes=10).to(device)

        train(student, train_loader, epochs=10, learning_rate=0.01, device=device)

        torch.save(teacher.state_dict(), "model/DeepNN_Adaptada.pth")
        torch.save(student.state_dict(), "model/student_no_kd_Adaptada.pth")

    else:
        teacher = load_model(DeepNN_Adaptada, "model/DeepNN_Adaptada.pth", device)
        student = load_model(LightNN_Adaptada, "model/student_no_kd_Adaptada.pth", device)

    teacher_params = "{:,}".format(sum(p.numel() for p in teacher.parameters()))
    print(f"Teacher Params: {teacher_params}")
    student_params = "{:,}".format(sum(p.numel() for p in student.parameters()))
    print(f"Student Params: {student_params}")
    test_accuracy_deep = test(teacher, test_loader, device)
    test_accuracy_light_ce = test(student, test_loader, device)
    print(f"Teacher accuracy: {test_accuracy_deep:.2f}%")
    print(f"Student accuracy: {test_accuracy_light_ce:.2f}%")

    # --- Experimento con distintos pesos alpha ---
    # alphas = [0.75]
    # Ts = [20]
    # results = {}

    # for alpha in alphas:
    #     results[alpha] = {}
    #     for T in Ts:
    #         new_student = LightNN_Adaptada(num_classes=10).to(device)
    #         print(f"\n=== Training student with alpha={alpha} / T={T} ===")
    #         train_knowledge_distillation(
    #             teacher=teacher,
    #             student=new_student.to(device),
    #             train_loader=train_loader,
    #             epochs=50,
    #             learning_rate=0.01,
    #             T=T,
    #             alpha=alpha,
    #             device=device
    #         )
    #         test_accuracy_light_ce_and_kd = test(new_student, test_loader, device)
    #         results[alpha][T] = test_accuracy_light_ce_and_kd
    #         print(f"Student accuracy (alpha={alpha} / T={T}): {test_accuracy_light_ce_and_kd:.2f}%")
    #
    # print("\n=== Summary ===")
    # for alpha, temp_dict in results.items():
    #     for T, acc in temp_dict.items():
    #         print(f"α={alpha:.2f} / T={T} → Accuracy: {acc:.2f}%")


# ============================ COSAS WANDB ============================

    sweep_config = {
        'method': 'bayes',  # bayesian, random, grid
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'alpha': {
                'min': 0.1,
                'max': 0.9
            },
            'T': {
                'min': 1,
                'max': 20
            },
            'learning_rate': {
                'values': [0.001]
            },
            'epochs': {
                'values': [10]
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="Basic_Knowledge_Distillation_Reentrenando")
    wandb.agent(sweep_id, lambda:train_kd_wandb(teacher, student))
