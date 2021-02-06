import os

import config
import train
import eval


def _train(vocab_file_path=None, model_file_path=None):
    print('\nStarting the training process......\n')

    if vocab_file_path:
        source_vocab_path, code_vocab_path, nl_vocab_path = vocab_file_path
        print('Vocabulary will be built by given file path.')
        print('\tSource code vocabulary path:\t', os.path.join(config.vocab_dir, source_vocab_path))
        print('\tTokenized source code vocabulary path:\t', os.path.join(config.vocab_dir, code_vocab_path))
        print('\tCode comment vocabulary path:\t', os.path.join(config.vocab_dir, nl_vocab_path))
    else:
        print('Vocabulary will be built according to dataset.')

    if model_file_path:
        print('Model will be built by given state dict file path:', os.path.join(config.model_dir, model_file_path))
    else:
        print('Model will be created by program.')

    print('\nInitializing the training environments......\n')
    train_instance = train.Train(vocab_file_path=vocab_file_path, model_file_path=model_file_path)
    print('Environments built successfully.\n')
    print('Size of train dataset:', train_instance.train_dataset_size)

    source_oov_rate = 1 - train_instance.source_vocab_size / train_instance.origin_source_vocab_size
    code_oov_rate = 1 - train_instance.code_vocab_size / train_instance.origin_code_vocab_size
    nl_oov_rate = 1 - train_instance.nl_vocab_size / train_instance.origin_nl_vocab_size

    print('\nSize of source code vocabulary:', train_instance.origin_source_vocab_size,
          '->', train_instance.source_vocab_size)
    print('Source code OOV rate: {:.2f}%'.format(source_oov_rate * 100))
    print('\nSize of tokenized source code vocabulary:', train_instance.origin_code_vocab_size,
          '->', train_instance.code_vocab_size)
    print('Tokenized source code OOV rate: {:.2f}%'.format(code_oov_rate * 100))
    print('\nSize of code comment vocabulary:', train_instance.origin_nl_vocab_size, '->', train_instance.nl_vocab_size)
    print('Code comment OOV rate: {:.2f}%'.format(nl_oov_rate * 100))
    config.logger.info('Size of train dataset:{}'.format(train_instance.train_dataset_size))
    config.logger.info('Size of source code vocabulary: {} -> {}'.format(
        train_instance.origin_source_vocab_size, train_instance.source_vocab_size))
    config.logger.info('Source source code OOV rate: {:.2f}%'.format(source_oov_rate * 100))
    config.logger.info('Size of tokenized source code vocabulary: {} -> {}'.format(
        train_instance.origin_code_vocab_size, train_instance.code_vocab_size))
    config.logger.info('Source code OOV rate: {:.2f}%'.format(code_oov_rate * 100))
    config.logger.info('Size of code comment vocabulary: {} -> {}'.format(
        train_instance.origin_nl_vocab_size, train_instance.nl_vocab_size))
    config.logger.info('Code comment OOV rate: {:.2f}%'.format(nl_oov_rate * 100))

    if config.validate_during_train:
        print('\nValidate every', config.validate_every, 'batches and each epoch.')
        print('Size of validation dataset:', train_instance.eval_instance.dataset_size)
        config.logger.info('Size of validation dataset: {}'.format(train_instance.eval_instance.dataset_size))

    print('\nStart training......\n')
    config.logger.info('Start training.')
    best_model = train_instance.run_train()
    print('\nTraining is done.')
    config.logger.info('Training is done.')

    # writer = SummaryWriter('runs/CodePtr')
    # for _, batch in enumerate(train_instance.train_dataloader):
    #     batch_size = len(batch[0][0])
    #     writer.add_graph(train_instance.model, (batch, batch_size, train_instance.nl_vocab))
    #     break
    # writer.close()

    return best_model


def _test(model):
    print('\nInitializing the test environments......')
    test_instance = eval.Test(model)
    print('Environments built successfully.\n')
    print('Size of test dataset:', test_instance.dataset_size)
    config.logger.info('Size of test dataset: {}'.format(test_instance.dataset_size))

    config.logger.info('Start Testing.')
    print('\nStart Testing......')
    test_instance.run_test()
    print('Testing is done.')


if __name__ == '__main__':
    best_model_dict = _train()
    _test(best_model_dict)
    # _test(os.path.join('20200526_095918', 'best_epoch-2_batch-1000.pt'))
