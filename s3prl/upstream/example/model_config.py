from transformers import BartConfig


def load_config():
    config = BartConfig()
    config.num_hidden_layers = 6
    config.hidden_size = 768
    config.encoder_ffn_dim = 2048
    config.hidden_act = 'relu'
    config.pad_index = 103
    config.word_pred = 0.15
    config.is_train_wav2vec=False
    config.is_finetune_wav2vec=False
    config.batch_size = 4
    config.real_batch_size = 16
    config.output_path='./output/'
    # config.wav2vec_dir='/userhome/code/audio_pretrain/pretrain_models/wav2vec2-base-960h'
    config.wav2vec_dir='F:\OneDrive\Code\Python\\audio_pretrain\pretrain_models\wav2vec2-base-960h'
    return config