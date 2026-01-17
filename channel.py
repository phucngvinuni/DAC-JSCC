import torch
import torch.nn as nn

class Channel(nn.Module):
    def __init__(self, channel_type='AWGN', snr=10):
        if channel_type not in ['AWGN', 'Rayleigh', 'Ideal']:
            raise Exception('Unknown type of channel')
        super(Channel, self).__init__()
        self.channel_type = channel_type
        # Store SNR dB as a tensor for consistent device handling
        self.snr_db_tensor = torch.tensor(float(snr)) 
        self.current_h_complex = None

    def forward(self, z_hat_input: torch.Tensor) -> torch.Tensor:
        if z_hat_input.dim() not in {2, 3, 4}:
            raise ValueError('Input tensor must be 2D, 3D or 4D')

        z_hat = z_hat_input.clone()
        current_device = z_hat.device
        
        # Normalize dimensions to 4D for internal processing [(N), C, (H, W)]
        # If 2D (N, L), treat as (N, L, 1, 1)
        original_dim = z_hat.dim()
        if original_dim == 2:
            z_hat = z_hat.unsqueeze(-1).unsqueeze(-1)
        elif original_dim == 3:
            z_hat = z_hat.unsqueeze(0)


        snr_db = self.snr_db_tensor.to(current_device)

        # Assume encoder normalization ensures signal power is ~1.0
        assumed_signal_power_per_sample = torch.tensor(1.0, device=current_device)

        # Calculate noise power
        snr_linear = 10**(snr_db / 10.0)
        noise_power_linear_per_sample_tensor = assumed_signal_power_per_sample / snr_linear
        noise_std_tensor = torch.sqrt(noise_power_linear_per_sample_tensor)

        # Add AWGN noise
        noise = torch.randn_like(z_hat) * noise_std_tensor
        z_processed = z_hat

        # Handle Rayleigh Fading
        if self.channel_type == 'Rayleigh':
            h_real = torch.randn(z_hat.size(0), 1, 1, 1, device=current_device) * torch.sqrt(torch.tensor(0.5, device=current_device))
            h_imag = torch.randn(z_hat.size(0), 1, 1, 1, device=current_device) * torch.sqrt(torch.tensor(0.5, device=current_device))
            
            num_channels_total = z_hat.size(1)
            # For complex fading, we pair channels. If odd, this needs handling, but typical JSCC uses even 'c' or 2*c.
            # Here we assume even channels for simplicity as per reference.
            if num_channels_total % 2 != 0:
                 # Fallback or error if strictly complex simulation is needed. 
                 # For now, we will just apply magnitude fading if odd, or raise error.
                 # Reference implied even channels. Let's raise error to be safe.
                raise ValueError("Number of channels must be even for complex Rayleigh fading simulation.")
            
            z_real_part = z_hat[:, :num_channels_total//2, :, :]
            z_imag_part = z_hat[:, num_channels_total//2:, :, :]
            z_complex_equivalent = torch.complex(z_real_part, z_imag_part)

            h_complex = torch.complex(h_real, h_imag)
            self.current_h_complex = h_complex

            z_faded_complex = z_complex_equivalent * self.current_h_complex
            z_faded_real_part = z_faded_complex.real
            z_faded_imag_part = z_faded_complex.imag
            z_processed = torch.cat((z_faded_real_part, z_faded_imag_part), dim=1)

        elif self.channel_type == 'Ideal':
            if is_3d_input: return z_hat.squeeze(0)
            return z_hat

        z_noisy = z_processed + noise

        # Equalization for Rayleigh (CSIR - Channel State Information at Receiver assumed known)
        if self.channel_type == 'Rayleigh' and self.current_h_complex is not None:
            num_channels_noisy = z_noisy.size(1)
            z_noisy_real_part = z_noisy[:, :num_channels_noisy//2, :, :]
            z_noisy_imag_part = z_noisy[:, num_channels_noisy//2:, :, :]
            z_noisy_complex_equivalent = torch.complex(z_noisy_real_part, z_noisy_imag_part)

            # Zero-forcing / MMSE simple equalization
            z_equalized_complex = z_noisy_complex_equivalent / (self.current_h_complex + 1e-9)
            z_equalized_real_part = z_equalized_complex.real
            z_equalized_imag_part = z_equalized_complex.imag
            z_final_output = torch.cat((z_equalized_real_part, z_equalized_imag_part), dim=1)
        else:
            z_final_output = z_noisy

        if original_dim == 3:
            return z_final_output.squeeze(0)
        elif original_dim == 2:
             return z_final_output.squeeze(-1).squeeze(-1)
        return z_final_output

    def get_channel_params(self):
        return self.channel_type, self.snr_db_tensor.item()
