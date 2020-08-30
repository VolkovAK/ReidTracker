from scipy.optimize import linear_sum_assignment
import numpy as np

from Track import Track, Detection


class Tracker():
    def __init__(self,
                 db=None,
                 max_iou_distance=0.8,
                 max_cosine_distance=0.4,
                 max_lost_counter=300,
                 adulthood_age=8):

        self.tracks = []
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        self.max_lost_counter = max_lost_counter
        self.adulthood_age = adulthood_age

        self.next_id = 1
        self.embeddings_db_neighbours = 5
        self.db = db
        self.match_function = None



    def update(self, bboxes, features=None):
        if features is not None:
            assert len(features) == len(bboxes)
        detections = []
        for bbox, feature in zip(bboxes, features):
            det = Detection(bbox, feature=feature)
            detections.append(det)

        for track in self.tracks:
            track.predict()

        matched, unmatched_tracks, unmatched_detections = \
            self.match_function(self.tracks, detections)

        for match in matched:
            self.tracks[match[0]].correct(self.db, detections[match[1]]) 
        for tr_idx in unmatched_tracks:
            self.tracks[tr_idx].mark_missed()
        for det_idx in unmatched_detections:
            self.initiate_track(detections[det_idx])

        # try to find already existing track for each new confirmed
        for track in self.tracks:
            if track.track_id == -1 and track.status == TrackStatus.confirmed: # it's new confirmed track
                track_features = track.features[:track.available_features] # get it collected features
                restored_id = None
                if self.db.size() > self.embeddings_db_neighbours: # dumb check for minimum allowed number features in DB
                    restored_id = self.db.try_to_restore_track_id(track_features) # try to find matching track id
                if restored_id is not None: # it was found
                    for old_track in self.tracks:
                        if old_track.track_id == restored_id: # find that old track with the same features
                            old_hist_pos = old_track.available_history
                            new_hist_pos = old_hist_pos + track.available_history
                            old_track.history_position[old_hist_pos:new_hist_pos] = track.history_position[:track.available_history]  #add new history to old track
                            old_track.available_history += track.available_history
                            old_track.copy_position_state(track)

                    track.status = TrackStatus.deleted
                    track.track_id = restored_id
                else:
                    track.track_id = self.next_id
                    self.next_id += 1
                self.db.add(track_features, np.array([track.track_id] * track.available_features)) # add new features to this track id

        remove_index = 0
        output_list = []
        tracks_to_remove = []
        while remove_index < len(self.tracks):
            t = self.tracks[remove_index]
            if t.is_deleted():
                if t.track_id > -1:
                    tracks_to_remove.append(t.track_id) # mark tracks ID to delete in features DB

                self.tracks.pop(remove_index)
                continue
            if t.lost_counter < 5 and t.status == TrackStatus.confirmed:
                output_list.append(t)
            remove_index += 1

        if len(tracks_to_remove) > 0:
            self.db.remove(np.array(tracks_to_remove, dtype=np.int64))

        return output_list


    def initiate_track(self, detection):
        self.tracks.append(Track(detection,
                                 track_id=-1, # it will be assigned later so we will not increase track id for every false detection
                                 max_lost_counter=self.max_lost_counter,
                                 adulthood_age=self.adulthood_age))


    def feature_match(self, tracks, detections):  # routine for matching bbox + features
        embedding_tracks_inds = []
        not_embedding_tracks_inds = []
        for i, track in enumerate(tracks):
            if track.status == TrackStatus.confirmed:
                embedding_tracks_inds.append(i)
            else:
                not_embedding_tracks_inds.append(i)

        matched_nn, unmatched_tracks, unmatched_detections = self.nn_matching(tracks, detections, track_indices=embedding_tracks_inds)

        unmatched_tracks = list(unmatched_tracks) + list(not_embedding_tracks_inds)

        unmatched_tracks_nn = [t for t in unmatched_tracks if self.tracks[t].lost_counter >= 5]
        # IOU match only for those tracks, that have been seen in the last 5 frames,
        # otherwise extrapolation error becomes too high, so only embeddings can be used in matching
        tracks_inds_for_iou = [t for t in unmatched_tracks if self.tracks[t].lost_counter < 5]
        track_for_iou = [t for i, t in enumerate(tracks) if i in tracks_inds_for_iou]
        detections_for_iou = [d for i, d in enumerate(detections) if i in unmatched_detections]

        matched_iou, unmatched_tracks_iou, unmatched_detections = \
            self.iou_matching(track_for_iou, detections_for_iou, tracks_inds_for_iou, unmatched_detections)

        unmatched_tracks = list(set(list(unmatched_tracks_nn) + list(unmatched_tracks_iou)))
        matched = list(matched_nn) + list(matched_iou)

        return matched, unmatched_tracks, unmatched_detections


    def nn_matching(self, tracks, detections, track_indices=None, detection_indices=None):
        if track_indices is None:
            track_indices = np.arange(len(tracks))
        else:
            track_indices = np.array(track_indices)
        if detection_indices is None:
            detection_indices = np.arange(len(detections))
        else:
            detection_indices = np.array(detection_indices)

        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.


        if self.db.size() > self.embeddings_db_neighbours:
            embs = np.array([d.feature for d in detections])
            cost_matrix, selected_ids = self.db.cost_matrix(embs) # get cost matrix for features and track ids

            track_index_map = dict() # dictionary that maps Track ID to its index in Tracker
            for i, track in enumerate(tracks):
                if track.track_id in selected_ids:
                    track_pos = selected_ids.index(track.track_id)
                    track_index_map[track_pos] = i

            potential_matches = np.where(cost_matrix < self.max_cosine_distance)

            # filtering unreachable matches
            for det_i, tr_i in zip(*potential_matches):
                tr_i_inner = track_index_map[tr_i]
                last_track_pos_on_plan = tracks[tr_i_inner].history_position[tracks[tr_i_inner].available_history - 1]
                last_5_track_pos_on_plan = tracks[tr_i_inner].history_position[tracks[tr_i_inner].available_history - 5]
                if detections[det_i].position is not None:
                    det_pos_on_plan = detections[det_i].position
                else:
                    det_pos_on_plan = np.array(detections[det_i].bbox[:2], dtype=np.float32)

                # distance between detection and last saved track position on plan
                euclidian_distance = np.linalg.norm(last_track_pos_on_plan - det_pos_on_plan)
                # average speed - distance between last position and fifth position from the last
                average_speed = np.linalg.norm(last_track_pos_on_plan - last_5_track_pos_on_plan) / 5
                # multiply average speed and count of lost frames to estimate possible traveled distance
                # and add mean of bbox size parameters in case of small speed
                max_plan_distance = average_speed * tracks[tr_i_inner].lost_counter + np.mean(tracks[tr_i_inner].current_state[2:])
                # make track unreachable if it is too far away from detection
                if euclidian_distance > max_plan_distance:
                    cost_matrix[det_i, tr_i] = 1

            candidatates_ids, tracks_ids = linear_sum_assignment(cost_matrix)

            matched_tracks_pos = cost_matrix[candidatates_ids, tracks_ids] < self.max_iou_distance
            not_matched_tracks_pos = matched_tracks_pos == False

            if any(matched_tracks_pos):
                tracks_indices_matched = [track_index_map[i] for i in tracks_ids[matched_tracks_pos]]
                matched = np.c_[np.array(tracks_indices_matched),
                                detection_indices[candidatates_ids[matched_tracks_pos]]]
                unmatched_tracks = [ind for ind_pos, ind in enumerate(track_indices) if ind_pos not in tracks_indices_matched]
                unmatched_candidates = [ind for ind_pos, ind in enumerate(detection_indices) if ind_pos not in candidatates_ids[matched_tracks_pos]]

                selected_embeddings = detection_indices[candidatates_ids[matched_tracks_pos]]
                selected_track_ids = np.array(selected_ids, dtype=np.int64)[tracks_ids[matched_tracks_pos]]
                self.db.add(embs[selected_embeddings], selected_track_ids)
            else:
                matched = []
                unmatched_tracks = track_indices
                unmatched_candidates = detection_indices


            return matched, unmatched_tracks, unmatched_candidates
        else:
            return [], track_indices, detection_indices



    def iou_matching(self, tracks, detections, track_indices=None, detection_indices=None):

        if track_indices is None:
            track_indices = np.arange(len(tracks))
        else:
            track_indices = np.array(track_indices)
        if detection_indices is None:
            detection_indices = np.arange(len(detections))
        else:
            detection_indices = np.array(detection_indices)

        if len(detection_indices) == 0 or len(track_indices) == 0:
            return [], track_indices, detection_indices  # Nothing to match.



        cost_matrix = calculate_iou_cost_matrix(tracks, detections, self.max_iou_distance)

        tracks_ids, candidatates_ids = linear_sum_assignment(cost_matrix)  # hungarian algorithm for optimal matching
        matched_tracks_pos = cost_matrix[tracks_ids, candidatates_ids] < self.max_iou_distance
        not_matched_tracks_pos = matched_tracks_pos == False

        if any(matched_tracks_pos):
            matched = np.c_[track_indices[tracks_ids[matched_tracks_pos]],
                            detection_indices[candidatates_ids[matched_tracks_pos]]]
            unmatched_tracks = [ind for ind_pos, ind in enumerate(track_indices) if ind_pos not in tracks_ids[matched_tracks_pos]]
            unmatched_candidates = [ind for ind_pos, ind in enumerate(detection_indices) if ind_pos not in candidatates_ids[matched_tracks_pos]]
        else:
            matched = []
            unmatched_tracks = track_indices
            unmatched_candidates = detection_indices


        return matched, unmatched_tracks, unmatched_candidates



def calculate_iou_cost_matrix(tracks, detections, iou_thresh=0.8): # speed up from vector version 4-6 times and more
    """Computer intersection over union.

    Parameters
    ----------
    bboxes : ndarray
        A matrix of candidate bounding boxes (one per row) in format
        `(center x, center y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bboxes`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the bboxes and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bboxes = np.array([track.current_state for track in tracks], dtype=np.float32)
    candidates = np.array([detection.bbox for detection in detections], dtype=np.float32)
    bboxes_tl = bboxes[:, :2] - bboxes[:, 2:]/2
    bboxes_br = bboxes[:, :2] + bboxes[:, 2:]/2
    candidates_tl = candidates[:, :2] - candidates[:, 2:]/2
    candidates_br = candidates[:, :2] + candidates[:, 2:]/2

    tl_br = np.array([
        [np.maximum(bboxes_tl[i, 0], candidates_tl[:, 0]),
        np.maximum(bboxes_tl[i, 1], candidates_tl[:, 1]),
        np.minimum(bboxes_br[i, 0], candidates_br[:, 0]),
        np.minimum(bboxes_br[i, 1], candidates_br[:, 1])] for i in range(len(bboxes))
    ])
    wh = np.maximum(0., tl_br[:, 2:, :] - tl_br[:, :2, :])

    area_intersection = wh.prod(axis=1)
    area_bboxes = bboxes[:, 2:].prod(axis=1)
    area_candidates = candidates[:, 2:].prod(axis=1)
    cost_matrix = 1 - np.array([area_intersection[i] / (area_bboxes[i] + area_candidates - area_intersection[i]) for i in range(len(bboxes))])
    cost_matrix[cost_matrix > iou_thresh] = 1
    return cost_matrix

