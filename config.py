batch_size = 8#256 * 8
batch_size_valid = batch_size

train_set_len = 180000*11 # for part 0 and for part 1: 655167
train_step_epoch = -(-train_set_len // batch_size)
# train_step_epoch = 10

valid_set_len = 50000
valid_step_epoch = -(-valid_set_len // batch_size_valid)
# valid_step_epoch = 10

args = {
    "train_dirs" : [
        'gs://kds-5d0f1271c081b5cd45a9299d83a2568db918e3e4d26ebd94f203f320',
        'gs://kds-42ce4442119b326b8339c043465e7247d8718f108023fb23cfc11c1c',
        'gs://kds-7110bce7bd75aed1574017a96b27c591e45488cb6f059e634a3d1318',
        'gs://kds-4244de11ef4f14801d6fa066d71dc5ed6e1c1ccea80270b31b4db07f',
        'gs://kds-b012012680947c49120003ea8e308f92f0a68a0537333f50390c92a2',
        'gs://kds-52a1383f8acbb3869de67709b011b4aa8e0e87f86df6a74ae38a3718',
        'gs://kds-126a803d35502776e8ca794919f5ca661314951255ceb0ae71bf17a8',
        'gs://kds-26cce936de2771226acd77ffe8954a787cd9b48bc02dbebf7afdf3c9',
        'gs://kds-9f49a97bc7acb4b160d5e4af634a991047a31b2064dff4999b5f3632',
        'gs://kds-fd421abab1d7a719c7d05fae05655924aead29e4862a62ee750b68c2',
        'gs://kds-4ce93939fa5debbdca757f22574d478d34cf95dc6fa224a0fb66663f'
        ],
    "valid_dirs" : [
        # "gs://kds-49298bb9b60e7369a0378148f07c8fd7c3671f304448c0a5a078dad0"
        ],
    "batch_size_train" : batch_size,
    "batch_size_valid" : batch_size_valid,
    "train_step_epoch" : train_step_epoch,
    "valid_step_epoch" : valid_step_epoch
}

