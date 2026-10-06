"""
Configuration for the Population-Aware ASD Classification framework.

The configuration contains the image, patch, representation,
population graph, GCN, Transformer, classifier, training,
cross-validation, and reproducibility settings used by the project.
"""

# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

SEED = 42


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

NUM_SUBJECTS = 1112
NUM_ASD = 539
NUM_TD = 573
NUM_SITES = 17


# ---------------------------------------------------------------------
# MRI preprocessing
# ---------------------------------------------------------------------

IMAGE_SIZE = 224
NUM_CHANNELS = 3


# ---------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------

PATCH_SIZE = 32
PATCH_GRID_SIZE = IMAGE_SIZE // PATCH_SIZE
NUM_PATCHES = PATCH_GRID_SIZE * PATCH_GRID_SIZE


# ---------------------------------------------------------------------
# Patch-level feature extraction
# ---------------------------------------------------------------------

PATCH_FEATURE_DIM = 512
RESNET_PRETRAINED = True


# ---------------------------------------------------------------------
# Subject-level representation
# ---------------------------------------------------------------------

EMBEDDING_DIM = 128
ATTENTION_HIDDEN_DIM = 128


# ---------------------------------------------------------------------
# Population graph
# ---------------------------------------------------------------------

GRAPH_SIMILARITY = "cosine"
GRAPH_KEEP_POSITIVE = True
GRAPH_USE_KNN = False
GRAPH_USE_FIXED_THRESHOLD = False
GRAPH_ADD_SELF_LOOPS = True


# ---------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------

GCN_LAYERS = 2
GCN_INPUT_DIM = EMBEDDING_DIM
GCN_HIDDEN_DIM = EMBEDDING_DIM
GCN_OUTPUT_DIM = EMBEDDING_DIM


# ---------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------

TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 4
TRANSFORMER_HIDDEN_DIM = EMBEDDING_DIM
TRANSFORMER_FFN_DIM = 256
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_MAX_NODES = NUM_SUBJECTS


# ---------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------

CLASSIFIER_INPUT_DIM = EMBEDDING_DIM
CLASSIFIER_HIDDEN_DIM = 64
CLASSIFIER_OUTPUT_DIM = 1
CLASSIFIER_DROPOUT = 0.3


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 10

GRADIENT_CLIP_MAX_NORM = 5.0


# ---------------------------------------------------------------------
# Learning-rate scheduler
# ---------------------------------------------------------------------

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5


# ---------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------

NUM_FOLDS = 10

STRATIFIED_CROSS_VALIDATION = True
SUBJECT_LEVEL_SPLIT = True

# The held-out fold is used only for final testing.
TEST_FOLD_AS_HELD_OUT = True


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

LOSS_FUNCTION = "binary_cross_entropy"


# ---------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------

OPTIMIZER = "Adam"


# ---------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------

MODEL_SELECTION_METRIC = "validation_accuracy"


# ---------------------------------------------------------------------
# Label definition
# ---------------------------------------------------------------------

TD_LABEL = 0
ASD_LABEL = 1
