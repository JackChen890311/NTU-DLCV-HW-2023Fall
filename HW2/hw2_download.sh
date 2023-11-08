#!/bin/bash
mkdir models
gdown -O models/model_2-1.pth '1xRxoNA5iNd2xF5uHVKVTvl5hYTMkn8r0'
gdown -O models/classifier_dann_DANN_mnistm_svhn.pt '1cmpNG6fGFX53xFIl4PFPRUA1hToqCgxx'
gdown -O models/classifier_dann_DANN_mnistm_usps.pt '1_Z9FR7NGCRd84cQ6UHxKI4QPhwLhfuzv'
gdown -O models/encoder_dann_DANN_mnistm_svhn.pt '1gUnsRrWZfRFxxrQbXQpGY5aiWXc6Ju6B'
gdown -O models/encoder_dann_DANN_mnistm_usps.pt '1CBgeXg2l7WCw0fA4qcaBcmgsX7q71ms9'