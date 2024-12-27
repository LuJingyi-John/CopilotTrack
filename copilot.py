import torch
import torch.nn.functional as F
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline


class CoTrackerThreeRealTime(CoTrackerThreeOnline):
    """Real-time point tracking implementation with online point addition capability.
    
    This class extends CoTrackerThreeOnline to provide real-time point tracking functionality
    with the ability to dynamically add new points to track during inference.
    
    Attributes:
        is_first_frame (bool): Flag indicating if current frame is first in sequence
        fmaps_pyramid (list): Multi-scale feature pyramid for tracking window
            Each level has shape (B, window_len, C, H_i, W_i) where H_i, W_i are
            feature dimensions at pyramid level i
        track_feat_support_pyramid (list): Feature support for tracked points at each pyramid level
            Each level contains tensor of shape (B, 1, corr_radius * corr_radius, N, C)
        coords (torch.Tensor): Tracked point coordinates in feature map scale, shape (B, window_len, N, 2)
        vis (torch.Tensor): Point visibility logits, shape (B, window_len, N, 1)
        conf (torch.Tensor): Tracking confidence logits, shape (B, window_len, N, 1)
    """

    def __init__(self, **args):
        """Initialize tracker with given parameters and reset states."""
        super().__init__(**args)
        self.reset()

    def reset(self):
        """Reset all tracking states for new sequence.
        
        Resets frame counter, feature pyramids, coordinate tensors and tracking logits
        to prepare for tracking a new sequence of frames.
        """
        self.is_first_frame = True
        self.fmaps_pyramid = None 
        self.track_feat_support_pyramid = None 
        self.coords = None
        self.vis = None
        self.conf = None

    def forward(self, frame, queries=None, iters=4, add_space_attn=True):
        """Process new frame and update tracking states.
        
        Args:
            frame (torch.Tensor): Input frame tensor of shape (B,3,H,W), values in range [0,255]
            queries (torch.Tensor, optional): New points to track of shape (B,N,2) in resolution scale
            iters (int): Number of refinement iterations for point locations. Defaults to 4
            add_space_attn (bool): Whether to use spatial attention. Defaults to True
            
        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor): 
                - Tracked point coordinates in image scale (B,N,2)
                - Point visibility probabilities (B,N,1)
                - Tracking confidence probabilities (B,N,1)
                Returns (None, None, None) if no points are being tracked
                
        Raises:
            ValueError: If input dimensions are invalid or incompatible with model stride
        """
        B, _, H, W = frame.shape
        device = frame.device
        window_len = self.window_len

        # Validate input dimensions and tracking state
        if not all([H % self.stride == 0, W % self.stride == 0, 
                   window_len >= 2]):
            raise ValueError("Invalid input parameters")

        # Do nothing when no points tracked
        if self.coords is None and queries is None:
            return None, None, None

        # Normalize frame and extract features, shape (B, latent_dim, H/stride, W/stride)
        frame = 2 * (frame / 255.0) - 1.0  # Scale to [-1,1]
        new_frame_fmap = F.normalize(self.fnet(frame), p=2, dim=1).to(frame.dtype)

        if self.is_first_frame:
            # First frame: Build multi-scale feature pyramid with sequential 2x downsampling
            self.fmaps_pyramid = []
            curr_fmap = new_frame_fmap
            for i in range(self.corr_levels):
                pyramid_level = curr_fmap if i == 0 else F.avg_pool2d(curr_fmap, 2, stride=2)
                curr_fmap = pyramid_level
                self.fmaps_pyramid.append(pyramid_level.unsqueeze(1).expand(B, window_len, -1, -1, -1))
                
            # Initialize tracking for first frame points
            if queries is not None:
                B, N, _ = queries.shape
                queries = queries / self.stride  # Scale to feature space
                
                # Initialize tracking states
                self.vis = torch.ones((B, window_len, N, 1), device=device) * 10
                self.conf = torch.ones((B, window_len, N, 1), device=device) * 10
                self.coords = queries.reshape(B, 1, N, 2).expand(B, window_len, N, 2)
                
                # Extract and store feature support at each pyramid level
                self.track_feat_support_pyramid = []
                for i in range(self.corr_levels):
                    _, track_feat_support = self.get_track_feat(
                        self.fmaps_pyramid[i],
                        torch.zeros(B, N, device=device).fill_(window_len-1),
                        queries / (2**i),  # Scale queries for pyramid level
                        support_radius=self.corr_radius
                    )
                    # Shape: (B, 1, corr_radius * corr_radius, N, latent_dim)
                    self.track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))
                    
            self.is_first_frame = False
            
        else:
            # Subsequent frames: Roll feature pyramid and update latest frame
            for i in range(len(self.fmaps_pyramid)):
                self.fmaps_pyramid[i] = torch.roll(self.fmaps_pyramid[i], shifts=-1, dims=1)
                self.fmaps_pyramid[i][:, -1] = (
                    F.avg_pool2d(new_frame_fmap, 2**i, stride=2**i) if i > 0 else new_frame_fmap
                )

            # Update tracking window by rolling previous states
            self.coords = torch.roll(self.coords, shifts=-1, dims=1)
            self.coords[:, -1] = self.coords[:, -2].clone()  # Initialize with previous positions
            
            self.vis = torch.roll(self.vis, shifts=-1, dims=1)
            self.vis[:, -1] = self.vis[:, -2].clone()  # Maintain visibility
            
            self.conf = torch.roll(self.conf, shifts=-1, dims=1)
            self.conf[:, -1] = 0  # Reset confidence for new frame

            # Refine point locations using correlation and attention
            coords, viss, confs = self.forward_window(
                fmaps_pyramid=self.fmaps_pyramid,
                coords=self.coords,
                track_feat_support_pyramid=self.track_feat_support_pyramid,
                vis=self.vis,
                conf=self.conf,
                iters=iters,
                add_space_attn=add_space_attn
            )

            # Store refined tracking states
            self.coords = coords[-1] / self.stride
            self.vis = viss[-1].unsqueeze(-1)
            self.conf = confs[-1].unsqueeze(-1)

            # Add new points if provided
            if queries is not None:
                B, N, _ = queries.shape
                queries = queries / self.stride  # Scale to feature space
                
                # Concatenate with existing states
                self.coords = torch.cat([
                    self.coords,
                    queries.reshape(B, 1, N, 2).expand(B, window_len, N, 2)
                ], dim=2)
                self.vis = torch.cat([
                    self.vis,
                    torch.ones((B, window_len, N, 1), device=device) * 10
                ], dim=2)
                self.conf = torch.cat([
                    self.conf,
                    torch.ones((B, window_len, N, 1), device=device) * -10
                ], dim=2) # Low confidence for new points in past frames
                self.conf[:, -1, -N:] = 10  # High confidence for new points in current frame

                # Update feature support pyramid
                for i in range(self.corr_levels):
                    _, track_feat_support = self.get_track_feat(
                        self.fmaps_pyramid[i],
                        torch.zeros(B, N, device=device).fill_(window_len-1),
                        queries / (2**i),
                        support_radius=self.corr_radius
                    )
                    support_expanded = track_feat_support.unsqueeze(1)
                    self.track_feat_support_pyramid[i] = torch.cat([
                        self.track_feat_support_pyramid[i],
                        support_expanded
                    ], dim=3)
            
        return (
            self.coords[:, -1] * self.stride,  # Scale coordinates back to image space
            torch.sigmoid(self.vis[:, -1]),    # Convert visibility logits to probabilities
            torch.sigmoid(self.conf[:, -1])    # Convert confidence logits to probabilities
        )

    @torch.no_grad()
    def warm_up(self):
        """Warm up model with random inputs to initialize states and cache.
        
        Performs two forward passes with random data to initialize model states
        and CUDA kernels, which can help reduce latency for first real inference.
        """
        device = next(self.parameters()).device
        random_frame = torch.rand(1, 3, *self.model_resolution, device=device)
        random_queries = torch.rand(1, 5, 2, device=device) * torch.tensor(self.model_resolution, device=device)
        
        self.reset()
        self(frame=random_frame, queries=random_queries, iters=1)
        self(frame=torch.rand_like(random_frame), iters=1)
        self.reset()
        
        torch.cuda.empty_cache()


class CoTrackerRealTimePredictor(torch.nn.Module):
    """Real-time point tracking predictor wrapping CoTracker model.
    
    This class provides a high-level interface for real-time point tracking,
    handling resolution scaling and visibility determination.
    
    Attributes:
        model (CoTrackerThreeRealTime): Underlying tracking model
        interp_shape (tuple): Target resolution (H,W) for model input
        visibility_threshold (float): Threshold for determining point visibility
    """
    
    def __init__(self, checkpoint="./checkpoints/scaled_online.pth", window_len=16):
        """Initialize predictor with model weights and parameters.
        
        Args:
            checkpoint (str): Path to model weights file. Set to None to skip loading weights
            window_len (int): Temporal window length for tracking. Defaults to 16
            
        Note:
            Model weights should be either a state dict or contain a 'model' key
            with the state dict
        """
        super().__init__()
        
        # Initialize tracking model
        self.model = CoTrackerThreeRealTime(
            stride=4,
            corr_radius=3,
            window_len=window_len
        )
        
        # Load model weights if provided
        if checkpoint:
            state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
            state_dict = state_dict.get("model", state_dict)
            self.model.load_state_dict(state_dict)
            print('Model weights loaded successfully.')
        
        # Set model constants
        self.interp_shape = self.model.model_resolution  # (H,W) for model input
        self.visibility_threshold = 0.6  # Threshold for point visibility

    def warm_up(self):
        """Warm up model with random inputs to initialize states."""
        self.model.warm_up()

    def reset(self):
        """Reset tracker states for new sequence."""
        self.model.reset()

    @torch.no_grad()
    def forward(self, frame, queries=None):
        """Process video frame and update point tracks.
        
        Args:
            frame (torch.Tensor): Input video frame of shape (B,C,H,W)
            queries (torch.Tensor, optional): Query points of shape (B,N,2) in original resolution
            
        Returns:
            tuple(torch.Tensor, torch.Tensor):
                - tracks: Tracked point coordinates (B,N,2) in original resolution
                - visibles: Boolean visibility flags (B,N,1) for tracked points
                Returns (None, None) for first frame with no queries
                
        Note:
            - Coordinates are scaled between model and original resolutions
            - Visibility is determined by combining visibility and confidence scores
        """
        # Get input dimensions
        H, W = frame.shape[-2:]
        model_h, model_w = self.interp_shape

        # Step 1: Resize frame to model resolution
        frame_resized = F.interpolate(
            frame, (model_h, model_w),
            mode="bilinear", align_corners=True
        )
        
        # Step 2: Scale queries to model resolution if provided
        if queries is not None:
            scale_to_model = torch.tensor([
                (model_w - 1) / (W - 1),
                (model_h - 1) / (H - 1)
            ]).to(queries.device)
            queries = queries * scale_to_model
            
        # Step 3: Run tracking model
        tracks, vis, conf = self.model(
            frame=frame_resized,
            queries=queries,
            iters=1,  # Single refinement iteration for real-time performance
            add_space_attn=True  # Enable spatial attention
        )

        # Return None for first frame with no queries
        if tracks is None:
            return None, None
            
        # Step 4: Scale coordinates back to original resolution
        scale_to_orig = torch.tensor([
            (W - 1) / (model_w - 1),
            (H - 1) / (model_h - 1)
        ]).to(tracks.device)
        tracks = tracks * scale_to_orig
        
        # Step 5: Compute visibility using confidence threshold
        visibles = (vis * conf) > self.visibility_threshold
        
        return tracks, visibles