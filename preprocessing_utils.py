import numpy as np

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def get_center_point_of_landmark(landmark_embeddings, left_bodypart, right_bodypart):
  left_x_y = landmark_embeddings[:, KEYPOINT_DICT[left_bodypart], :2]
  right_x_y = landmark_embeddings[:, KEYPOINT_DICT[right_bodypart], :2]
  center_x_y = (left_x_y + right_x_y) / 2
  return center_x_y

def get_pose_size(landmark_embeddings, torso_size_multiplier=2.5):
  # hips center
  hips_center_x_y = get_center_point_of_landmark(landmark_embeddings, 'left_hip', 'right_hip')

  # shoulders center
  shoulders_center_x_y = get_center_point_of_landmark(landmark_embeddings, 'left_shoulder', 'right_shoulder')

  # torso length as the minimum body size
  torso_size = np.linalg.norm(shoulders_center_x_y - hips_center_x_y, axis=1, keepdims=True)

  # pose center
  pose_center = hips_center_x_y
  pose_center = np.repeat(pose_center, 17, axis=0).reshape(landmark_embeddings.shape[0], -1, 2)
  distance_to_pose_center = np.linalg.norm(landmark_embeddings - pose_center, axis=2, keepdims=True)

  # max dist to pose center
  max_dist = np.amax(distance_to_pose_center, axis=1)

  # normalize scale
  pose_size = np.maximum(torso_size * torso_size_multiplier, max_dist)

  return pose_size

def normalize_landmark_embeddings(landmark_embeddings):
  landmark_embeddings_centers = get_center_point_of_landmark(landmark_embeddings, 'left_hip', 'right_hip')
  landmark_embeddings_centers = np.repeat(landmark_embeddings_centers, 17, axis=0).reshape(landmark_embeddings.shape[0], -1, 2) # reshape
  landmark_embeddings_centered = landmark_embeddings - landmark_embeddings_centers
  pose_size = get_pose_size(landmark_embeddings_centered)
  return landmark_embeddings_centered / pose_size[:, None]

def process_landmark_embeddings(landmark_embeddings):
  landmark_embeddings = landmark_embeddings.reshape(-1, 17, 3) # reshape
  landmark_embeddings = landmark_embeddings[:, :, :2] # drop score

  # normalize landmarks
  landmark_embeddings_normalized = normalize_landmark_embeddings(landmark_embeddings)

  # flatten the normalized landmark coordinates into a vector
  landmark_embeddings_normalized = landmark_embeddings_normalized.reshape(landmark_embeddings_normalized.shape[0], -1, 1)

  return landmark_embeddings_normalized[:, :, 0]