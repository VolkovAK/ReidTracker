import numpy as np
import time


class Detection():
    def __init__(self, bbox, position=None, feature=None):
        self.bbox = bbox
        self.position = np.array(position, dtype=np.float32) if position is not None else None
        self.feature = np.array(feature, dtype=np.float32) if feature is not None else None


class TrackStatus():
    tentative = 0
    confirmed = 1
    deleted = 2



class Track():
    def __init__(self,
                 detection,
                 track_id,
                 max_features=20,
                 max_history=10000,
                 max_lost_counter=3000,
                 max_kid_lost_counter=2,
                 adulthood_age=8):
        # x_center, y_center, w, h (bboxes)
        self.current_state = np.array(detection.bbox, dtype=np.float32) # текущие параметры ббокса
        self.velocity_state = np.zeros((4), dtype=np.float32) # скорость изменений параметров ббокса
        self.observation = np.array(detection.bbox, dtype=np.float32) # наблюдаемые (новые) параметры ббокса
        self.state_weight = np.array([0.3, 0.3, 0.5, 0.5], dtype=np.float32) # веса изменений параметров. Чем меньше, тем меньше учитывается текущие состояние и тем больше влияет новое
        self.observation_weight = 1 - self.state_weight
        self.observation_velocity = np.zeros((4), dtype=np.float32) # изменение между наблюдаемыми параметрами (новым и старым)

        self.history_position = np.zeros((max_history, 2), dtype=np.float32) # история позиций 
        self.history_position[0] = detection.bbox[:2]
        self.available_history = 1
        self.max_history = max_history

        self.features = np.zeros((max_features, len(detection.feature)), dtype=np.float32) # эмбеддинги (временное хранилище, до матчинга)
        self.features[0] = detection.feature
        self.available_features = 1
        self.max_features = max_features
        self.current_feature = detection.feature

        self.track_id = -1  # Номер трэка, получаемый после взросления (status == confirmed)

        self.lost_counter = 0
        self.age = 1
        self.adulthood_age = adulthood_age
        self.max_kid_lost_counter = max_kid_lost_counter
        self.max_lost_counter = max_lost_counter
        self.status = TrackStatus.tentative

        self.lifetime_start = time.time()
        self.lifetime = 0
        self.lifetime_last = 0


    def to_tlwh(self):
        tlwh = self.current_state.copy()
        tlwh[:2] -= tlwh[2:]//2
        return tlwh

    def get_lifetime_str(self):
        minutes = int(self.lifetime // 60)
        seconds = int(self.lifetime % minutes)
        return '{:02}:{:02}'.format(minutes, seconds)

    def copy_position_state(self, copy_from):
        self.current_state = copy_from.current_state
        self.velocity_state = copy_from.velocity_state
        self.observation = copy_from.observation
        self.state_weight = copy_from.state_weight
        self.observation_weight = copy_from.observation_weight
        self.observation_velocity = copy_from.observation_velocity
        self.lost_counter = 0
        self.age = copy_from.age


    def predict(self):
        # x = x + vx, ...
        self.current_state = self.current_state + self.velocity_state
        self.lost_counter += 1
        self.lifetime = time.time() - self.lifetime_start


    def correct(self, embeddings_db, detection):
        new_observation = np.array(detection.bbox) # update last observation with new information

        if self.lost_counter < 10: # we have seen this object not too long ago, smoothing can be applied
            new_observation_velocity = new_observation - self.observation
            rand_observation_velocity = np.abs((new_observation_velocity - self.observation_velocity)) / (np.abs((new_observation_velocity + self.observation_velocity)) + 0.0001)

            self.observation = new_observation
            self.observation_velocity = new_observation_velocity

            self.state_weight = self.state_weight + np.tanh(rand_observation_velocity - 1) / 5
            self.state_weight = np.where(self.state_weight < 0.05, 0.05, self.state_weight)
            self.state_weight = np.where(self.state_weight > 0.8, 0.8, self.state_weight)
            self.observation_weight = 1 - self.state_weight

            # update current states
            self.current_state = self.observation * self.observation_weight + self.current_state * self.state_weight
            self.velocity_state = self.velocity_state * self.state_weight + (self.observation - self.current_state) * self.observation_weight
        else: # last time we have seen this track is so deep in the past, that we are even started to nostalgize
            self.observation = new_observation
            self.observation_velocity = np.zeros((4), dtype=np.float32)
            self.current_state = new_observation
            self.velocity_state = np.zeros((4), dtype=np.float32)


        # update features
        if self.status == TrackStatus.tentative:
            self.features[self.available_features] = detection.feature
            self.available_features += 1
	if self.status == TrackStatus.confirmed:
           self.current_feature = detection.feature


        # update history
        if self.available_history >= self.max_history:
            self.available_history -= 1
            self.history_position[:self.max_history-1] = self.history_position[1:]
        self.history_position[self.available_history] = new_observation[:2]
        self.available_history += 1

        self.lost_counter = 0
        self.age += 1
        if self.status == TrackStatus.tentative and self.age >= self.adulthood_age:
            self.status = TrackStatus.confirmed

        self.lifetime_last = time.time()


    def mark_missed(self):
        if (self.status == TrackStatus.tentative and self.lost_counter > self.max_kid_lost_counter) or self.lost_counter > self.max_lost_counter:
            self.status = TrackStatus.deleted

    def is_deleted(self):
        return self.status == TrackStatus.deleted


