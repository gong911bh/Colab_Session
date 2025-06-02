def train_model(model, train_data, train_labels=None, epochs=10, batch_size=32,
                criterion=None, optimizer=None, lr=1e-3):
    import tensorflow as tf
    import torch

    tf_gpu_available = tf.config.list_physical_devices('GPU')
    torch_gpu_available = torch.cuda.is_available()

    if isinstance(model, tf.keras.Model):
        device = "GPU" if tf_gpu_available else "CPU"
        print(f"[INFO] Detected TensorFlow model. Using {device}.")

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size)

    elif isinstance(model, torch.nn.Module):
        device = torch.device("cuda" if torch_gpu_available else "cpu")
        print(f"[INFO] Detected PyTorch model. Using {device}.")
        model.to(device)
        model.train()

        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(train_data, train_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        if criterion is None:
            criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    else:
        raise TypeError("Check the model type!")

def predict_model(model, input_data):
    import tensorflow as tf
    import torch
    import numpy as np

    if isinstance(model, tf.keras.Model):
        return model.predict(input_data)

    elif isinstance(model, torch.nn.Module):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float()
        else:
            input_tensor = input_data

        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
... 
...         input_tensor = input_tensor.to(device)
... 
...         with torch.no_grad():
...             output = model(input_tensor)
... 
...         return output.cpu().numpy()
... 
...     else:
...         raise TypeError("Check the model type!")
... 
>>> def evaluate_model(model, test_data, test_labels, criterion=None):
...     import tensorflow as tf
...     import torch
...     import numpy as np
... 
...     if isinstance(model, tf.keras.Model):
...         print("[INFO] Evaluating TensorFlow model...")
...         return model.evaluate(test_data, test_labels, verbose=0)
... 
...     elif isinstance(model, torch.nn.Module):
...         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
...         model.to(device)
...         model.eval()
... 
...         if isinstance(test_data, np.ndarray):
...             x_tensor = torch.from_numpy(test_data).float()
...         else:
...             x_tensor = test_data
... 
...         if isinstance(test_labels, np.ndarray):
...             y_tensor = torch.from_numpy(test_labels).long()
...         else:
...             y_tensor = test_labels
... 
...         if len(x_tensor.shape) == 3:
...             x_tensor = x_tensor.unsqueeze(0)
...         if len(y_tensor.shape) == 0:
...             y_tensor = y_tensor.unsqueeze(0)
... 
...         x_tensor, y_tensor = x_tensor.to(device), y_tensor.to(device)
... 
...         if criterion is None:
...             criterion = torch.nn.CrossEntropyLoss()
... 
...         with torch.no_grad():
...             outputs = model(x_tensor)
...             loss = criterion(outputs, y_tensor)
...             preds = torch.argmax(outputs, dim=1)
...             accuracy = (preds == y_tensor).float().mean().item()
... 
...         return loss.item(), accuracy
... 
...     else:
