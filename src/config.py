import argparse
import numpy as np
import ast


def get_config(argv=None):
    parser = argparse.ArgumentParser()

    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='funnel', choices=['funnel', 'mg25', 'diamond', 'multimodal_swissroll'], help='Dataset to use')
    parser.add_argument('--rep_size', type=int, default=5, help='Number of the repetitions')
    parser.add_argument('--d', type=int, default=100, help='Data dimensionality')
    parser.add_argument('--n_samples', type=int, default=50000, help='Number of training samples')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')

    # Model parameters
    parser.add_argument('--mid_features', type=int, default=512, help='Hidden width of the score network')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of temporal layers')

    # SDE parameters
    parser.add_argument('--a', nargs='+', action=ParseList, default=[0.1, 0.25, 0.5, 1, 2], help='Drift coefficient for CLD')
    parser.add_argument('--epsilon', nargs='+', action=ParseList, default=[0.0, 0.1, 0.25, 0.5, 1], help='Position-noise regularization (0: vanilla CLD; >0: regularized)')
    parser.add_argument('--sigma', type=float, default=float(np.sqrt(2.0)), help='Velocity noise scale')
    parser.add_argument('--T', type=float, default=1.0, help='Final time horizon')
    parser.add_argument('--beta_min', type=float, default=4.0, help='Minimum β value (schedule lower bound)')
    parser.add_argument('--beta_max', type=float, default=4.0, help='Maximum β value (schedule upper bound)')
    parser.add_argument('--v0_var', type=float, default=0.04 * 0.25, help='Initial velocity variance')
    parser.add_argument('--numerical_eps', type=float, default=1e-6, help='Small jitter added to covariances for numerical stability')

    # Training parameters
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping max norm')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay for parameter averaging')
    parser.add_argument('--n_epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--loss_eps', type=float, default=1e-5, help='Lower bound used in the loss-time sampling')

    # Sampling parameters
    parser.add_argument('--num_steps', type=int, default=1000, help='Number of discretization steps for samplers')
    parser.add_argument('--sampling_eps', type=float, default=1e-3, help='Minimum time epsilon for sampling schedule')
    parser.add_argument('--sampling_size', type=int, default=50000, help='Number of samples to draw during evaluation')

    # Evaluation parameters
    parser.add_argument('--eval_interval', type=int, default=100, help='Evaluate the score every N epochs')

    return parser.parse_args(argv)


class ParseList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        s = " ".join(values)
        try:
            if s.lstrip().startswith('['):                
                xs = ast.literal_eval(s)
                if not isinstance(xs, (list, tuple)): raise ValueError
            else:                                         
                xs = s.replace(',', ' ').split()
            setattr(namespace, self.dest, [float(x) for x in xs])
        except Exception:
            raise argparse.ArgumentError(self, f"Invalid --value: {s!r}")
