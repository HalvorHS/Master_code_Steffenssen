from tmu.preprocessing.DNA_data_pre import preprocess_genome_data

if __name__ == "__main__":
    main_folder = (r'C:\Users\hhste\OneDrive - Norwegian University of Life '
                   r'Sciences\Documents\skole\Master\Tsetlin_maskin\Tsetlin_code\data')
    processed_data = preprocess_genome_data(main_folder)
    print(processed_data)
