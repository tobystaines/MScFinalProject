import datetime
import os
import errno
import math
import tensorflow as tf


def train(sess, model, model_config, model_folder, handle, training_iterator, training_handle, validation_iterator,
          validation_handle, writer):
    """
    Train an audio_models.py model.
    """

    def validation(last_val_cost, min_val_cost, min_val_check, worse_val_checks, model, val_check):
        """
        Perform a validation check using the validation dataset.
        """
        print('Validating')
        sess.run(validation_iterator.initializer)
        val_costs = list()
        val_iteration = 1
        print('{ts}:\tEntering validation loop'.format(ts=datetime.datetime.now()))
        while True:
            try:
                val_cost = sess.run(model.cost, {model.is_training: False, handle: validation_handle})
                if val_iteration % 200 == 0:
                    print("{ts}:\tValidation iteration: {i}, Loss: {vc}".format(ts=datetime.datetime.now(),
                                                                                i=val_iteration, vc=val_cost))
                if math.isnan(val_cost):
                    print('Error: cost = nan\nDiscarding batch')
                else:
                    val_costs.append(val_cost)
                    val_iteration += 1
            except tf.errors.OutOfRangeError:
                # Calculate and record mean loss over validation dataset
                val_check_mean_cost = sum(val_costs) / len(val_costs)
                print('Validation check mean loss: {l}'.format(l=val_check_mean_cost))
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='Validation_mean_loss', simple_value=val_check_mean_cost)])
                writer.add_summary(summary, val_check)
                # If validation loss has worsened increment the counter, else, reset the counter
                if val_check_mean_cost > last_val_cost:
                    worse_val_checks += 1
                    print('Validation loss has worsened. worse_val_checks = {w}'.format(w=worse_val_checks))
                else:
                    worse_val_checks = 0
                    print('Validation loss has improved!')
                if val_check_mean_cost < min_val_cost:
                    min_val_cost = val_check_mean_cost
                    print('New best validation cost!')
                    min_val_check = val_check
                last_val_cost = val_check_mean_cost

                break

        return last_val_cost, min_val_cost, min_val_check, worse_val_checks

    def checkpoint(model_config, model_folder, saver, sess, global_step):
        """
        Take a checkpoint of the model.
        """
        # Make sure there is a folder to save the checkpoint in
        checkpoint_path = os.path.join(model_config["model_base_dir"], model_folder)
        try:
            os.makedirs(checkpoint_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        print('Checkpoint')
        saver.save(sess, os.path.join(checkpoint_path, model_folder), global_step=int(global_step))
        return os.path.join(checkpoint_path, model_folder + '-' + str(global_step))

    print('Starting training')
    # Initialise variables and define summary
    epoch = 0
    iteration = 1
    last_val_cost = 1
    min_val_cost = 1
    min_val_check = None
    val_check = 1
    worse_val_checks = 0
    latest_checkpoint_path = os.path.join(model_config['model_base_dir'], model_config['checkpoint_to_load'])

    cost_summary = tf.summary.scalar('Training_loss', model.cost)
    if model_config['data_type'] == 'mag':
        mix_summary = tf.summary.image('Mixture', model.mixed_input)
        voice_summary = tf.summary.image('Voice', model.voice_input)
        mask_summary = tf.summary.image('Voice_Mask', model.voice_mask)
        gen_voice_summary = tf.summary.image('Generated_Voice', model.gen_voice)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, write_version=tf.train.SaverDef.V2)
    sess.run(training_iterator.initializer)

    # Begin training loop
    # Train for the specified number of epochs, unless early stopping is triggered
    while epoch < model_config['epochs'] and worse_val_checks < model_config['num_worse_val_checks']:
        try:
            if model_config['data_type'] == 'mag':
                try:
                    _, cost, cost_sum, mix, voice, mask, gen_voice = sess.run([model.train_op, model.cost, cost_summary,
                                                                               mix_summary, voice_summary, mask_summary,
                                                                               gen_voice_summary], {model.is_training: True,
                                                                                                    handle: training_handle})
                except RuntimeWarning:
                    print('Invalid value encountered. Ignoring batch.')
                    continue
            else:
                try:
                    _, cost, cost_sum = sess.run([model.train_op, model.cost, cost_summary],
                                             {model.is_training: True, handle: training_handle})
                except RuntimeWarning:
                    print('Invalid value encountered. Ignoring batch.')
                    continue
            if math.isnan(cost):
                print('Error: cost = nan')
                print('Loading latest checkpoint')
                restorer = tf.train.Saver()
                restorer.restore(sess, latest_checkpoint_path)
                break
            writer.add_summary(cost_sum, iteration)  # Record the loss at each iteration
            if iteration % 200 == 0:
                print("{ts}:\tTraining iteration: {i}, Loss: {c}".format(ts=datetime.datetime.now(),
                                                                         i=iteration, c=cost))
            # If saving by iterations, take a checkpoint
            if model_config['saving'] and not model_config['save_by_epochs'] and iteration % model_config['save_iters'] == 0:
                latest_checkpoint_path = checkpoint(model_config, model_folder, saver, sess, iteration)

            # If using early stopping by iterations, enter validation loop
            if model_config['early_stopping'] and not model_config['val_by_epochs'] and iteration % model_config['val_iters'] == 0:
                last_val_cost, min_val_cost, min_val_check, worse_val_checks = validation(last_val_cost,
                                                                                          min_val_cost,
                                                                                          min_val_check,
                                                                                          worse_val_checks,
                                                                                          model,
                                                                                          val_check)
                val_check += 1

            iteration += 1

        # When the dataset is exhausted, note the end of the epoch
        except tf.errors.OutOfRangeError:
            print('{ts}:\tEpoch {e} finished after {i} iterations.'.format(ts=datetime.datetime.now(),
                                                                           e=epoch, i=iteration))
            if model_config['data_type'] == 'mag':
                try:
                    writer.add_summary(mix, iteration)
                    writer.add_summary(voice, iteration)
                    writer.add_summary(mask, iteration)
                    writer.add_summary(gen_voice, iteration)
                except NameError:  # Indicates the try has not been successfully executed at all
                    print('No images to record')
                    break
            epoch += 1
            # If using early stopping by epochs, enter validation loop
            if model_config['early_stopping'] and model_config['val_by_epochs'] and iteration > 1:
                last_val_cost, min_val_cost, min_val_check, worse_val_checks = validation(last_val_cost,
                                                                                          min_val_cost,
                                                                                          min_val_check,
                                                                                          worse_val_checks,
                                                                                          model,
                                                                                          val_check)
                val_check += 1
            if model_config['saving'] and model_config['save_by_epochs']:
                latest_checkpoint_path = checkpoint(model_config, model_folder, saver, sess, epoch)
            sess.run(training_iterator.initializer)
    print('Training complete after {e} epochs.'.format(e=epoch))
    if model_config['early_stopping'] and worse_val_checks >= model_config['num_worse_val_checks']:
        print('Stopped early due to validation criteria.')
    else:
        # Final validation check
        if (iteration % model_config['val_iters'] != 1 or not model_config['early_stopping']) and not model_config['val_by_epochs']:
            last_val_cost, min_val_cost, min_val_check, _ = validation(last_val_cost, min_val_cost, min_val_check,
                                                                       worse_val_checks, model, val_check)
    print('Finished requested number of epochs.')
    print('Final validation loss: {lvc}'.format(lvc=last_val_cost))
    if last_val_cost == min_val_cost:
        print('This was the best validation loss achieved')
    else:
        print('Best validation loss ({mvc}) achieved at validation check {mvck}'.format(mvc=min_val_cost,
                                                                                        mvck=min_val_check))
    return model

