#!/usr/bin/env python3
"""Kaggle push-up preprocessing. Run from pose_training_offline/scripts' parent.
Visual source groups are provisional, not verified subject identities.
No existing datasets or caches are modified.
"""
from pathlib import Path
import argparse, json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import pandas as pd

WORK=Path("data/preprocess/kaggle_pushup")
CACHE=Path("data/cache/kp_full/kaggle_pushup_yolov8m")
OUT=Path("results/pushup/kaggle_preprocess")
EXPECTED_WEIGHTS="dbe539ea268db2534390942cfdf206e521f376f19e5415967a57f6a2ddfa3c90"
RECORDS=json.loads(r"""
[
 {
  "sample_id": "KPU_001",
  "relative_path": "Wrong sequence/1.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "8e84056ba27365e3b5f0407e010d628b5c6721169e65a848f0af56e9c11ff5b6",
  "frames": 290,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_002",
  "relative_path": "Correct sequence/Copy of push up 1.mp4",
  "group": "S02_gray_timer",
  "label": 1,
  "sha256": "d31d35d19cfe657fe86aa57a110956c05bb622fb4479e48c9ce2c216abf2e467",
  "frames": 61,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_003",
  "relative_path": "Wrong sequence/2.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "a566b5a947a3300341ed830503c3924e9d3a473defc20e65789484d1edfe8001",
  "frames": 158,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_004",
  "relative_path": "Correct sequence/Copy of push up 2.mp4",
  "group": "S02_gray_timer",
  "label": 1,
  "sha256": "2f21d2a270aebd1d50d6dd181824c4225123f8f51c51cb673dab7a81de1723ca",
  "frames": 54,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_005",
  "relative_path": "Wrong sequence/3.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "f2fcf9a88f9ebf2233722d6ea9f0de694a90d1a1c48721f8b37de34fd385d47a",
  "frames": 104,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_006",
  "relative_path": "Correct sequence/Copy of push up 3.mp4",
  "group": "S02_gray_timer",
  "label": 1,
  "sha256": "15325c91f633f0bd3e4432610d8877d137a45118fc04515673c22fcce0cdb441",
  "frames": 52,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_007",
  "relative_path": "Wrong sequence/4.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "43f8465b813b332b1bb79c7bc7e24605dc65ba429acf844b2e8d5bd4156ef5a0",
  "frames": 72,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_008",
  "relative_path": "Correct sequence/Copy of push up 4.mp4",
  "group": "S02_gray_timer",
  "label": 1,
  "sha256": "3a60977f2e1a6903039de79445936c0478611c4f59f9d786eae931da7762e732",
  "frames": 57,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_009",
  "relative_path": "Wrong sequence/5.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "9ed5dec3113e333b07a6643158f785c9a14780b3c538e0292bf6b00b9f4d458a",
  "frames": 119,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_010",
  "relative_path": "Correct sequence/Copy of push up 5.mp4",
  "group": "S02_gray_timer",
  "label": 1,
  "sha256": "d8c796cb5faaf17845209b41c4e4b534c8bdf9dd1a235297c5d139010c159be9",
  "frames": 53,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_011",
  "relative_path": "Wrong sequence/6.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "d8a2d6a94f2b0fa5a61368786bb899d36cd2853a640ece37359157ee7597398b",
  "frames": 119,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_012",
  "relative_path": "Correct sequence/Copy of push up 6.mp4",
  "group": "S02_gray_timer",
  "label": 1,
  "sha256": "069d9887d5c838df68114f047872d342a6fa287c37cb282ab014fc93350a8ceb",
  "frames": 60,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_013",
  "relative_path": "Wrong sequence/7.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "bc0425b2ad2e29d0c31d4acf0e36b57fcf458baba5bebf68757dd736524cd1b6",
  "frames": 110,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_014",
  "relative_path": "Correct sequence/Copy of push up 7.mp4",
  "group": "S02_gray_timer",
  "label": 1,
  "sha256": "4bc6dfc0ecf311304b32a0d78f121eee946f484e3b0e615e7b39847a07aa5728",
  "frames": 58,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_015",
  "relative_path": "Wrong sequence/8.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "86e06da20f1cbd4bc9291e3ff3a309df5deaf87eeb2441717fe22eed975c7593",
  "frames": 87,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_016",
  "relative_path": "Wrong sequence/9.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "601cec0c61a90321516e7d9febc60203eb967ae5b55731cea528458b82be32d1",
  "frames": 66,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_017",
  "relative_path": "Wrong sequence/10.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "00a5373577f3892faf92af61e014cf058c597aa1ed3560b9bca28b617dd2d2f1",
  "frames": 126,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_018",
  "relative_path": "Wrong sequence/11.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "9574ba002f6143e49f6b50be288c6fc9cc3334b8d50b7c969b95217a0caabc86",
  "frames": 83,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_019",
  "relative_path": "Wrong sequence/12.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "c9bd8b268d479b8fed02b6dee2fbbc1acae399e3c438323e46afec6abf2443f1",
  "frames": 121,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_020",
  "relative_path": "Wrong sequence/13.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "ee88ff49de224543e453f796fbcfc01ca48d9d570f5bc55d35deeb575ef67437",
  "frames": 121,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_021",
  "relative_path": "Wrong sequence/14.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "d4d024bc45f09f6c9cc6f87e72c435caaee0895d68687f04411ef754e80e399b",
  "frames": 125,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_022",
  "relative_path": "Wrong sequence/15.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "ab600bef17cf514f3de2f3b58285350d8b8e1db515d585d1f63c2254c503bef9",
  "frames": 100,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_023",
  "relative_path": "Wrong sequence/16.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "a8d1dca38a6636d16f0d8e7cfc0326226e75f9bc5801835f54bcc3938628f5a7",
  "frames": 100,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_024",
  "relative_path": "Wrong sequence/17.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "fb84187163e1357b528aa7e959c65f9f40543937f5f0b6c2ae282d9808db93aa",
  "frames": 107,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_025",
  "relative_path": "Wrong sequence/18.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "7766b38f6da092cfe3755bb789fbf36e5e064a9655699cdb212db1d9357915af",
  "frames": 153,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_026",
  "relative_path": "Wrong sequence/19.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "1e042b1f296ad24f335539a2dcbf2376a7271b8192ff94b45363d67165af5aa4",
  "frames": 126,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_027",
  "relative_path": "Wrong sequence/20.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "282b0f10627a0aedfb75f6a15554c340b4f090e7f12c605b049d1aa00d54ab86",
  "frames": 93,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_028",
  "relative_path": "Wrong sequence/21.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "63d6dbfcce3cb2e7aa8f5e2ef9f54a296823f0c091edf8790a1297af9f049ee8",
  "frames": 105,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_029",
  "relative_path": "Wrong sequence/22.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "0c08d600d8881fcc2d224d60f056253852398032a4870851345c2754af70a2a3",
  "frames": 250,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_030",
  "relative_path": "Wrong sequence/23.mp4",
  "group": "S01_carpet_sofa",
  "label": 0,
  "sha256": "66e75f75f58030c3090da5c43f7222151b8af2303ab1b0fbea910d76e8dca0dc",
  "frames": 128,
  "fps": 29.97
 },
 {
  "sample_id": "KPU_031",
  "relative_path": "Wrong sequence/Copy of push up 36.mp4",
  "group": "S03_concrete_room",
  "label": 0,
  "sha256": "1b14b60be6ed551259d62aa4a963de7bb22387e7915f2a8857f48905bed7ff13",
  "frames": 83,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_032",
  "relative_path": "Wrong sequence/Copy of push up 37.mp4",
  "group": "S03_concrete_room",
  "label": 0,
  "sha256": "9b50997dd5dc11e21ad5c0758f9c6c261d20df94dd77601e5f07464a017d4f31",
  "frames": 66,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_033",
  "relative_path": "Wrong sequence/Copy of push up 42.mp4",
  "group": "S04_gym_white_shirt",
  "label": 0,
  "sha256": "d31caa4a2b160554e8694543e685f5733e75fb5867d781c00bd2d172933a4107",
  "frames": 116,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_034",
  "relative_path": "Wrong sequence/Copy of push up 44.mp4",
  "group": "S04_gym_white_shirt",
  "label": 0,
  "sha256": "78bd83ade4e930dea567d81da33fdf6009ec8cdcc29e5b1b4338253116bd121f",
  "frames": 99,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_035",
  "relative_path": "Wrong sequence/Copy of push up 46.mp4",
  "group": "S04_gym_white_shirt",
  "label": 0,
  "sha256": "40ab55f05a18f02e36d1e122eef4bcc15c25130ac3b856e45c347e7aece69423",
  "frames": 100,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_036",
  "relative_path": "Correct sequence/Copy of push up 47.mp4",
  "group": "S04_gym_white_shirt",
  "label": 1,
  "sha256": "432d676b8fcd0e5d760421a1a3c57a684e5abc1117a69db8a4929160bedbc284",
  "frames": 95,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_037",
  "relative_path": "Wrong sequence/Copy of push up 55.mp4",
  "group": "S05_dark_blue_shorts",
  "label": 0,
  "sha256": "2748b2bf66945656d51d963efa4f9bcf46ffa965fa8438303b99b9ba016c3c70",
  "frames": 113,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_038",
  "relative_path": "Wrong sequence/Copy of push up 56.mp4",
  "group": "S05_dark_blue_shorts",
  "label": 0,
  "sha256": "d07da9efe843ee91d86517318eae3bd6dde62d857e047f15e772d0723b8812e3",
  "frames": 107,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_039",
  "relative_path": "Wrong sequence/Copy of push up 57.mp4",
  "group": "S05_dark_blue_shorts",
  "label": 0,
  "sha256": "4068546107645e34711484109bab3503f309963b9d990c9ec72b4d58c086a98f",
  "frames": 75,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_040",
  "relative_path": "Wrong sequence/Copy of push up 58.mp4",
  "group": "S05_dark_blue_shorts",
  "label": 0,
  "sha256": "28ee658acb7f70156987d3d6d281bb903cf1059bcd072431dc3a14fa50dda549",
  "frames": 84,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_041",
  "relative_path": "Correct sequence/Copy of push up 78.mp4",
  "group": "S06_orange_clothing",
  "label": 1,
  "sha256": "301ebc48a95c364a3824ea78a3299773162ef039db80e7e42603cda2885d6342",
  "frames": 58,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_042",
  "relative_path": "Correct sequence/Copy of push up 79.mp4",
  "group": "S06_orange_clothing",
  "label": 1,
  "sha256": "d1b08e851503adb7e30a56572bc20750f8ac7f2691bc476de96e43b2b5f288b2",
  "frames": 45,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_043",
  "relative_path": "Correct sequence/Copy of push up 80.mp4",
  "group": "S07_turquoise_top",
  "label": 1,
  "sha256": "3c12c49fdbc75ac8145d121348a974cece6cf47de9794c0e43ca12e815409bfa",
  "frames": 157,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_044",
  "relative_path": "Wrong sequence/Copy of push up 81.mp4",
  "group": "S08_gym_blue_top",
  "label": 0,
  "sha256": "67d5c4c422da284197ea68e0f0f00b64d1a764efaf24816b867ab80184f5fe70",
  "frames": 93,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_045",
  "relative_path": "Correct sequence/Copy of push up 83.mp4",
  "group": "S08_gym_blue_top",
  "label": 1,
  "sha256": "fa1efae60ce73598172014da8ade83be76da7812fd3d1877fe3d8435db69eeac",
  "frames": 63,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_046",
  "relative_path": "Correct sequence/Copy of push up 85.mp4",
  "group": "S08_gym_blue_top",
  "label": 1,
  "sha256": "739762426a3eb27c7af6b1cc81287d4fd28eb3d8dc0394bd23b71e1e9f771522",
  "frames": 61,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_047",
  "relative_path": "Correct sequence/Copy of push up 100.mp4",
  "group": "S09_window_mat",
  "label": 1,
  "sha256": "1060cc9ce19051611570a531cce919fb38af041142773cb201fc24d6834056ba",
  "frames": 124,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_048",
  "relative_path": "Correct sequence/Copy of push up 101.mp4",
  "group": "S09_window_mat",
  "label": 1,
  "sha256": "a4691de76584097c3cdb5a015b2b5ca84f655de03c4358fa9f58a864f77229ec",
  "frames": 77,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_049",
  "relative_path": "Correct sequence/Copy of push up 102.mp4",
  "group": "S09_window_mat",
  "label": 1,
  "sha256": "b4d26e91446d5146c672c000980ab3e262831342e796614c5b090b1f0a835bf2",
  "frames": 83,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_050",
  "relative_path": "Wrong sequence/Copy of push up 104.mp4",
  "group": "S09_window_mat",
  "label": 0,
  "sha256": "1211f3c41e5cd82b66fa20f87c925ea4d796a0c2db8fcfe5b67e0cbea755d414",
  "frames": 232,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_051",
  "relative_path": "Wrong sequence/Copy of push up 105.mp4",
  "group": "S09_window_mat",
  "label": 0,
  "sha256": "d5483ffc6abc75876a412d000e2f8aaaabc6c490a017367cff536109ca30b0c1",
  "frames": 230,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_052",
  "relative_path": "Wrong sequence/Copy of push up 107.mp4",
  "group": "S09_window_mat",
  "label": 0,
  "sha256": "cff638188a84c3d654a61ce953ea2e6e871fa7997c02832a9c62bc329ad7486e",
  "frames": 81,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_053",
  "relative_path": "Wrong sequence/Copy of push up 108.mp4",
  "group": "S09_window_mat",
  "label": 0,
  "sha256": "1201a71e74b64cdc144fa3398068285c6618ab68bba6e4ed4a08dea5dd529f5c",
  "frames": 230,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_054",
  "relative_path": "Wrong sequence/Copy of push up 110.mp4",
  "group": "S09_window_mat",
  "label": 0,
  "sha256": "b7414d13a180f5b94b67e6ea1ce42eb54c8e62b5a81c771d5f7a1bb89a6d4eb0",
  "frames": 212,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_055",
  "relative_path": "Correct sequence/Copy of push up 113.mp4",
  "group": "S09_window_mat",
  "label": 1,
  "sha256": "33398e7546f01668c14f6b463156df1bfcb9bad2dedfeb89cb2a8fd6ec240e6b",
  "frames": 77,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_056",
  "relative_path": "Correct sequence/Copy of push up 114.mp4",
  "group": "S09_window_mat",
  "label": 1,
  "sha256": "80767ea7190ecfc9f39556bf90596b1c39c39aec4737554d0651a96ffe362a41",
  "frames": 88,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_057",
  "relative_path": "Correct sequence/Copy of push up 115.mp4",
  "group": "S09_window_mat",
  "label": 1,
  "sha256": "c210d09bb98e2fd04a28c9516fdd37f7c1c786d6ec54c4df6bfadbf731b88464",
  "frames": 137,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_058",
  "relative_path": "Correct sequence/Copy of push up 116.mp4",
  "group": "S10_weight_rack",
  "label": 1,
  "sha256": "7164bb7d80dee275fad574c6b94775950f162a710df6e4f3d22c588a826b0563",
  "frames": 75,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_059",
  "relative_path": "Correct sequence/Copy of push up 127.mp4",
  "group": "S11_grass",
  "label": 1,
  "sha256": "3a5a8d4179f88467e47f87981ab75b2bbc1eda4c4366780eaa507748f519be9b",
  "frames": 91,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_060",
  "relative_path": "Correct sequence/Copy of push up 129.mp4",
  "group": "S11_grass",
  "label": 1,
  "sha256": "0d516ee84fc5b5d972dd48b5f6e0e3d80372224bbda84cf8786f1ef35b1beba9",
  "frames": 84,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_061",
  "relative_path": "Correct sequence/Copy of push up 131.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "78fc0a86e4fea4d66a6f524998768d9ac3b36729da4f6a083f9f8e36b42fee73",
  "frames": 77,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_062",
  "relative_path": "Correct sequence/Copy of push up 132.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "889fc8047cabce4e33503eef5a860a4a822bb861196484816365f892f91d33f1",
  "frames": 79,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_063",
  "relative_path": "Correct sequence/Copy of push up 133.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "d3501ff82b6ea12d3dae64c86ab0b8a9c5af7ab6fda898387de4f25e6992047f",
  "frames": 76,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_064",
  "relative_path": "Correct sequence/Copy of push up 134.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "bef886134ea0f284d8e8372ae41b7030ee62a02b25fdd5847b74b24fca1ba76f",
  "frames": 75,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_065",
  "relative_path": "Correct sequence/Copy of push up 135.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "d350d0f2491f76e2e0eee8f3c77c8c5105517481bc241cea046a8382b35b8413",
  "frames": 72,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_066",
  "relative_path": "Correct sequence/Copy of push up 136.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "24a0d7cfead441493c1a1b26fd193d8d4f32885596ab3f28362ca67abb37988a",
  "frames": 75,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_067",
  "relative_path": "Correct sequence/Copy of push up 137.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "f9f0e9de93ec30ebb1440d9982528b75210c31ef609e21fc2e63dc01871d1a52",
  "frames": 71,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_068",
  "relative_path": "Correct sequence/Copy of push up 138.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "826375b5bb3b53f8c5f3a1097363573750f77a55b8d7bd9649abda5117dd4c26",
  "frames": 74,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_069",
  "relative_path": "Correct sequence/Copy of push up 140.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "43d6598474987fe726991bac4d309a250cdf21e30db9598c4cb7c5dc1be6dbc4",
  "frames": 87,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_070",
  "relative_path": "Correct sequence/Copy of push up 141.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "601361122eae4ff53c82d92d9e886e91a10d07134ede5ea1f220d229d3c65736",
  "frames": 91,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_071",
  "relative_path": "Correct sequence/Copy of push up 142.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "f09c803877be22520af799422fc7c38a7348fb7e1e966c6e156905dfaf1a7473",
  "frames": 92,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_072",
  "relative_path": "Correct sequence/Copy of push up 144.mp4",
  "group": "S12_blue_lockers",
  "label": 1,
  "sha256": "9e7ae7cffe031d0bb61b76bc065f0362a7b9f7ded8da4ed0162ad473eac1118e",
  "frames": 90,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_073",
  "relative_path": "Correct sequence/Copy of push up 146.mp4",
  "group": "S13_glass_wall",
  "label": 1,
  "sha256": "99260c208fa90a5f480a0c4adf97f256836be301ef0d4f0747f4cf01379a4564",
  "frames": 107,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_074",
  "relative_path": "Correct sequence/Copy of push up 147.mp4",
  "group": "S13_glass_wall",
  "label": 1,
  "sha256": "691c3551e027fa6341faabc7238a615e0940fac69b7c5d9a78cbf58e619199fd",
  "frames": 123,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_075",
  "relative_path": "Correct sequence/Copy of push up 148.mp4",
  "group": "S13_glass_wall",
  "label": 1,
  "sha256": "8b5f7380ac5870763bc464fc26cda452c26bc603e0a444a4a87a16e6f1bb1897",
  "frames": 133,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_076",
  "relative_path": "Wrong sequence/Copy of push up 149.mp4",
  "group": "S14_home_mat",
  "label": 0,
  "sha256": "391cb1eb3ebc429478342bd3f1739c6b2fa08b161076269e6c143e4685ae9a12",
  "frames": 324,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_077",
  "relative_path": "Wrong sequence/Copy of push up 150.mp4",
  "group": "S15_dark_gym_red_rack",
  "label": 0,
  "sha256": "99e2e8d4f1960ba313f697bda239221abf410a0a36d0dff72be4c7af79f66cba",
  "frames": 63,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_078",
  "relative_path": "Wrong sequence/Copy of push up 151.mp4",
  "group": "S15_dark_gym_red_rack",
  "label": 0,
  "sha256": "f6ceac501f67a5fdbcd61e6ecc85e4b43a6cdf63405b77cc489b5991298968f6",
  "frames": 62,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_079",
  "relative_path": "Wrong sequence/Copy of push up 152.mp4",
  "group": "S15_dark_gym_red_rack",
  "label": 0,
  "sha256": "9bd5f3cba72097ffc42145cbe6a2200ccbc6974276e2de113a27cea746eacc65",
  "frames": 66,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_080",
  "relative_path": "Wrong sequence/Copy of push up 153.mp4",
  "group": "S15_dark_gym_red_rack",
  "label": 0,
  "sha256": "eda4679175b0f2ad838163c155cd996c722f954cd569962640c45a3f701ce7f1",
  "frames": 64,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_081",
  "relative_path": "Wrong sequence/Copy of push up 155.mp4",
  "group": "S15_dark_gym_red_rack",
  "label": 0,
  "sha256": "be26a742674c34ff759c1d56ee88f907ca0a55075849503587ba414c6b7a01f9",
  "frames": 60,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_082",
  "relative_path": "Wrong sequence/Copy of push up 156.mp4",
  "group": "S15_dark_gym_red_rack",
  "label": 0,
  "sha256": "5eb5e0a5f3d96d4e2bafcafae15e9bc8f5b880bde87c1324e0e61fa29ca9bfa2",
  "frames": 47,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_083",
  "relative_path": "Correct sequence/Copy of push up 162.mp4",
  "group": "S16_white_wall_gray_floor",
  "label": 1,
  "sha256": "95d95e0c7a1cd31fb8af2fc60c5dcfa29acffdd5bda13d250357bc67b07ef44f",
  "frames": 107,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_084",
  "relative_path": "Correct sequence/Copy of push up 163.mp4",
  "group": "S16_white_wall_gray_floor",
  "label": 1,
  "sha256": "e8ea63aed2b14b39ed7887bcb1d3401d3d3aa5c8a912b054ce95f05ee4a58512",
  "frames": 98,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_085",
  "relative_path": "Correct sequence/Copy of push up 164.mp4",
  "group": "S16_white_wall_gray_floor",
  "label": 1,
  "sha256": "0212fb967e95401e6d99471a814539d373a4b9dd4890c6936adae6753fcb6f46",
  "frames": 115,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_086",
  "relative_path": "Correct sequence/Copy of push up 165.mp4",
  "group": "S16_white_wall_gray_floor",
  "label": 1,
  "sha256": "a86ee09693de37ec4b705c4309dae2d323123b2f5aea209d7fbb57a7cc30066b",
  "frames": 100,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_087",
  "relative_path": "Wrong sequence/Copy of push up 166.mp4",
  "group": "S16_white_wall_gray_floor",
  "label": 0,
  "sha256": "b08d0e327850fe8890d85dca3600c4daa29dcb9afa0491fea32abef2208bef05",
  "frames": 172,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_088",
  "relative_path": "Wrong sequence/Copy of push up 167.mp4",
  "group": "S16_white_wall_gray_floor",
  "label": 0,
  "sha256": "ab6b6c072c62d25c6525a02f74190a0379f3a978e8f1141550c19c48f884c107",
  "frames": 85,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_089",
  "relative_path": "Wrong sequence/Copy of push up 168.mp4",
  "group": "S16_white_wall_gray_floor",
  "label": 0,
  "sha256": "aff96d0bc56a131298297449e6ec578464e8cb7618ee659250b950b569357678",
  "frames": 90,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_090",
  "relative_path": "Wrong sequence/Copy of push up 171.mp4",
  "group": "S17_yellow_shirt_gym",
  "label": 0,
  "sha256": "54397990f4d7d580bd039417da1adbfa874898a8b21175071a2b54bd8c27a524",
  "frames": 397,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_091",
  "relative_path": "Correct sequence/Copy of push up 173.mp4",
  "group": "S17_yellow_shirt_gym",
  "label": 1,
  "sha256": "2c2784423820c28179a44b555b50612492a38f697ab944589d38cc2e66684315",
  "frames": 153,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_092",
  "relative_path": "Correct sequence/Copy of push up 174.mp4",
  "group": "S17_yellow_shirt_gym",
  "label": 1,
  "sha256": "fec77633294f01abdf3b82f053242509bd8e22430c7551684bfab3b8819831e9",
  "frames": 131,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_093",
  "relative_path": "Correct sequence/Copy of push up 175.mp4",
  "group": "S17_yellow_shirt_gym",
  "label": 1,
  "sha256": "1e46f0d0d7e847b6ce9adc7ddcfd6e2185ac9198b5583b8aadde1c5a3870ece4",
  "frames": 110,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_094",
  "relative_path": "Correct sequence/Copy of push up 177.mp4",
  "group": "S18_dark_gym_diagram",
  "label": 1,
  "sha256": "e5d489e4464c36aa6c748a5d9ec0c3795730fa1032a7c1368f953c7ddd8cbbf3",
  "frames": 146,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_095",
  "relative_path": "Correct sequence/Copy of push up 178.mp4",
  "group": "S18_dark_gym_diagram",
  "label": 1,
  "sha256": "8c2d03872d0fca65c8725b120ec0ac31477fe19d433966e14b57a30438bb426d",
  "frames": 148,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_096",
  "relative_path": "Correct sequence/Copy of push up 186.mp4",
  "group": "S19_white_instruction",
  "label": 1,
  "sha256": "3b0794814dcdaf5e535681c9c7c7f023db777c1ef4e45415f260468e1125736e",
  "frames": 68,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_097",
  "relative_path": "Correct sequence/Copy of push up 187.mp4",
  "group": "S19_white_instruction",
  "label": 1,
  "sha256": "32a8acbc782064b03924abb037871625b98dfd117d0bc535eff074fd6f31692e",
  "frames": 70,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_098",
  "relative_path": "Correct sequence/Copy of push up 188.mp4",
  "group": "S19_white_instruction",
  "label": 1,
  "sha256": "b3b18be5a8f6ea7ccbcc3bdaefd014f5defd9cafa5538ae2c7b4a88ab8947e60",
  "frames": 69,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_099",
  "relative_path": "Correct sequence/Copy of push up 197.mp4",
  "group": "S20_blue_shutter",
  "label": 1,
  "sha256": "ba22e90492034aa77bb87f751cd011f8317b4b51a5bb3edf241d282229f0d028",
  "frames": 156,
  "fps": 25.0
 },
 {
  "sample_id": "KPU_100",
  "relative_path": "Wrong sequence/Copy of push up 198.mp4",
  "group": "S20_blue_shutter",
  "label": 0,
  "sha256": "6180d8f7346f5b9e1103f83971346f49dac228da8d5750cc528225826e51161c",
  "frames": 211,
  "fps": 25.0
 }
]
""")

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode",choices=["all","auto","review","extract","audit"],default="all",nargs="?")
    ap.add_argument("--root",type=Path,default=Path.home()/"capstone2/datasets/Kaggle-Pushup")
    ap.add_argument("--weights",type=Path,default=Path.home()/"capstone2/pose_training/yolov8m-pose.pt")
    ap.add_argument("--sample",action="append")
    ap.add_argument("--no-open",action="store_true")
    args=ap.parse_args()
    import preprocess_rehab24 as pre
    import preprocess_extra_datasets as extra
    from extract_video_kp_cache import UltralyticsPoseBackend,sha256_file,KP_DUMP_DTYPE
    from eval_pushup_counter_pilot import PushupCounter
    from train_pushup_posture import extract_features,quality,FEATURE_NAMES

    # Reuse the established extractor and tracking policy, but isolate all outputs.
    extra.WORK=WORK
    if sha256_file(args.weights)!=EXPECTED_WEIGHTS:
        raise ValueError("Pose weights differ from the MV7 baseline")
    rows=[]
    actual={str(p.relative_to(args.root)) for p in args.root.rglob("*.mp4")}
    if actual!={r["relative_path"] for r in RECORDS}:
        raise ValueError("Video inventory changed from audited dataset")
    for r in RECORDS:
        p=args.root/r["relative_path"]
        if sha256_file(p)!=r["sha256"]: raise ValueError(f"Video hash changed: {p}")
        info=extra.metadata(p)
        if info["frames"]!=r["frames"] or abs(info["fps"]-r["fps"])>.002 or (info["width"],info["height"])!=(640,360):
            raise ValueError(f"Metadata changed: {p}")
        rows.append(dict(sample_id=r["sample_id"],video_id=r["sample_id"],
            dataset="kaggle_pushup",group_id="kpu:"+r["group"],source_group=r["group"],
            path=str(p.resolve()),cache_dir=str(CACHE.resolve()),clip_correctness=r["label"],
            exercise="push_up",label_scope="clip",annotations=[],sha256=r["sha256"],**info))
    WORK.mkdir(parents=True,exist_ok=True)
    pre.save_json(WORK/"manifest.json",dict(weights=str(args.weights.resolve()),
        weights_sha256=EXPECTED_WEIGHTS,videos=rows,group_note="Provisional visual source groups; not subject IDs"))
    selected_rows=[v for v in rows if not args.sample or v["sample_id"] in args.sample]
    if not selected_rows or args.sample and set(args.sample)!={v["sample_id"] for v in selected_rows}:
        raise ValueError("Unknown sample")
    p=WORK/"anchors.json"
    anchors=pre.load_json(p) if p.exists() else {}
    if args.mode in ("all","auto","review","extract"):
        backend=UltralyticsPoseBackend(str(args.weights.resolve()),device="0",imgsz=640,conf=.25)
        if args.mode in ("all","auto","review"):
            extra.auto_review(selected_rows,backend,anchors,args.mode=="review",args.no_open)
        if args.mode in ("all","extract"):
            extra.extract(selected_rows,backend,anchors,EXPECTED_WEIGHTS)
    if args.mode not in ("all","audit"): return
    # Always audit all 100 clips, even for a partial extraction.
    OUT.mkdir(parents=True,exist_ok=True)
    videos=[]; reps=[]; feats=[]
    for v in rows:
        sid=v["sample_id"]; fps=v["fps"]; n=v["frames"]
        row=dict(sample_id=sid,group=v["group_id"],clip_label=v["clip_correctness"],
                 frames=n,fps=fps,anchor_available=sid in anchors,candidates=0,accepted=0)
        state=pre.check_cache(v,EXPECTED_WEIGHTS,verify_video=True,verify_anchor=anchors.get(sid))
        row["cache_status"]=state
        if state!="complete" or sid not in anchors:
            videos.append(row); continue
        paths=pre.cache_paths(v)
        raw=np.memmap(paths[0],dtype=KP_DUMP_DTYPE,mode="r")
        with np.load(paths[2],allow_pickle=False) as z:
            status=z["status"]; boxes=z["selected_box"]
        if len(raw)!=n or len(status)!=n or len(boxes)!=n or not np.array_equal(raw["frame_id"],np.arange(n)):
            raise ValueError(f"Cache frame mismatch {sid}")
        sel=(status==0)&(raw["pose_valid"]!=0)
        kp=np.array(raw["kp"],dtype=np.float32,copy=True)
        kp[~sel,:,:2]=np.nan; kp[~sel,:,2]=0
        q,why=quality(kp,sel,fps)
        center=(boxes[:,:2]+boxes[:,2:])/2
        diag=np.maximum(np.linalg.norm(boxes[:,2:]-boxes[:,:2],axis=1),1)
        jump=(status[:-1]==0)&(status[1:]==0)&(np.linalg.norm(np.diff(center,axis=0),axis=1)/diag[:-1]>=.25)
        row.update(q,clip_quality_reasons=";".join(why),box_jumps=int(jump.sum()))
        c=PushupCounter(fps)
        for i,k in enumerate(kp):
            e=c.update(k,i,bool(sel[i]))
            if e is None: continue
            st,ed=int(e["start_frame"]),int(e["end_frame"])
            qrep,whyrep=quality(kp[st:ed+1],sel[st:ed+1],fps)
            if jump[max(0,st-1):min(len(jump),ed+1)].any():
                whyrep.append("box_jump_touching_rep")
            x=None
            if not whyrep:
                try: x=extract_features(kp[st:ed+1])
                except ValueError as exc: whyrep.append("feature:"+str(exc))
            accepted=x is not None
            r=dict(sample_id=sid,group=v["group_id"],clip_label=v["clip_correctness"],
                   **e,**qrep,accepted=accepted,reasons=";".join(whyrep),feature_row=-1)
            if accepted:
                r["feature_row"]=len(feats); feats.append(x); row["accepted"]+=1
            row["candidates"]+=1; reps.append(r)
        videos.append(row)
        print(sid,state,"candidates",row["candidates"],"accepted",row["accepted"],flush=True)
    vd=pd.DataFrame(videos)
    rd=pd.DataFrame(reps)
    vd.to_csv(OUT/"per_video.csv",index=False)
    rd.to_csv(OUT/"candidate_reps.csv",index=False)
    X=np.asarray(feats,dtype=float).reshape(-1,len(FEATURE_NAMES))
    np.save(OUT/"features.npy",X)
    counts=vd.groupby("clip_label")[["candidates","accepted"]].sum()
    bylabel={str(int(i)): {k:int(v) for k,v in r.items()} for i,r in counts.iterrows()}
    summary=dict(clips=len(vd),source_groups=int(vd.group.nunique()),
        cache_status={str(k):int(v) for k,v in vd.cache_status.value_counts().items()},
        anchors=int(vd.anchor_available.sum()),candidate_total=int(vd.candidates.sum()),
        accepted_total=int(vd.accepted.sum()),by_label=bylabel,
        clips_with_accepted=int((vd.accepted>0).sum()),
        groups_with_accepted=int(vd.loc[vd.accepted>0,"group"].nunique()),
        feature_shape=list(X.shape),feature_names=FEATURE_NAMES,
        feature_source_sha256=sha256_file(Path(__file__).with_name("train_pushup_posture.py")),
        counter_source_sha256=sha256_file(Path(__file__).with_name("eval_pushup_counter_pilot.py")),
        features_sha256=sha256_file(OUT/"features.npy"),
        candidate_reps_sha256=sha256_file(OUT/"candidate_reps.csv"),
        note="No manual GT counts or boundaries. Candidate clips can start/end mid-rep. Clip labels are weak rep labels. No training performed.")
    pre.save_json(OUT/"summary.json",summary)
    with ZipFile(OUT/"kaggle_preprocess.zip","w",ZIP_DEFLATED) as z:
        for name in ("summary.json","per_video.csv","candidate_reps.csv"):
            z.write(OUT/name,name)
        for name in ("manifest.json","manual_review_queue.json"):
            if (WORK/name).exists(): z.write(WORK/name,name)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
