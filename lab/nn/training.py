import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import ChessPositionEvaluationsDataset
from network import ChessEvaluationCNN


def main():
    plt.switch_backend('Agg')

    chess_dataset = ChessPositionEvaluationsDataset()

    train_dataset, validation_dataset, _ = random_split(
        chess_dataset,
        # [0.001, 0.001, 0.998],
        [0.01, 0.01, 0.98],
        generator=torch.Generator().manual_seed(42)
    )
    data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True,
                             persistent_workers=True)
    val_data_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True,
                                 persistent_workers=True)
    # # Windows
    # data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True,
    #                          persistent_workers=False)
    # val_data_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True,
    #                              persistent_workers=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChessEvaluationCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1_000
    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, batch in enumerate(data_loader):
            board_tensor = batch['board_tensor'].to(device)
            active_player = batch['active_player'].to(device)
            half_move_clock = batch['half_move_clock'].to(device)
            target = batch['evaluation'].to(device)

            optimizer.zero_grad()

            output = model(board_tensor, active_player, half_move_clock)
            output = output.squeeze(1)

            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epoch_loss = running_loss / len(data_loader)
        train_losses.append(epoch_loss)

        # Validation phase
        model.eval()

        with torch.no_grad():
            val_batch = next(iter(val_data_loader))
            val_board_tensor = val_batch['board_tensor'].to(device)
            val_active_player = val_batch['active_player'].to(device)
            val_half_move_clock = val_batch['half_move_clock'].to(device)
            val_target = val_batch['evaluation'].to(device)

            val_outputs = model(val_board_tensor, val_active_player, val_half_move_clock)
            val_outputs = val_outputs.squeeze(1)
            val_batch_loss = criterion(val_outputs, val_target)

            val_batch_loss_item = val_batch_loss.item()
            validation_losses.append(val_batch_loss_item)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, ' +
                  f'Validation Loss: {val_batch_loss_item:.4f}', end='\r')

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Losses up to Epoch {epoch + 1}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'losses_epoch.png')
        plt.close()

        torch.save(model.state_dict(), 'model.pth')


if __name__ == '__main__':
    main()
