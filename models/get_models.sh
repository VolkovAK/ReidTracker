#!/bin/bash

cd "$(dirname "$0")"

wget https://download.01.org/opencv/openvino_training_extensions/models/person_reidentification/person-reidentification-retail-0300.pt -O reid300.pt
wget https://download.01.org/opencv/openvino_training_extensions/models/person_reidentification/person-reidentification-retail-0249.pt -O reid249.pt

