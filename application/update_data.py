from modules.interface.update_data.store_embed_reg_merged import generate_embeddings_file
import pickle
import os

def update():
    with open('modules/interface/data/all_analogies_df_mini', 'rb') as f:
        # Index(['source_language', 'target_language', 'A', 'B', 'C', 'D'], dtype='object')
        data = pickle.load(f)

    voc = set(data['A'].values)
    voc.update(set(data['B'].values))
    voc.update(set(data['C'].values))
    voc.update(set(data['D'].values))
    with open('modules/interface/data/full_voc', 'wb') as f:
        pickle.dump(voc, f)

    if os.path.isfile('modules/nn_analogy_solver/embeddings/all_vectors.txt.pt'):
        os.remove('modules/nn_analogy_solver/embeddings/all_vectors.txt.pt')
    if os.path.isfile('modules/nn_analogy_solver/embeddings/all_vectors.txt'):
        os.remove('modules/nn_analogy_solver/embeddings/all_vectors.txt')

    path_models = 'modules/nn_analogy_solver/models/omnilingual_solver.pth'
    custom_embeddings_file = 'modules/nn_analogy_solver/embeddings/all_vectors.txt'
    generate_embeddings_file(path_embed = path_models,
                              storing_path = custom_embeddings_file,
                              emb_size = 512,
                              full_dataset=True)

if __name__ == "__main__":
    update()
