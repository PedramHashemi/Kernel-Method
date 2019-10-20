import pandas as pd
from sklearn.impute import SimpleImputer
from keras.layers import Input, Dense, Dropout
from keras.models import Model


def reader(expressionMatrixFile, geneExpressionFile):
    # READ THE FILES
    gExp = pd.read_csv(geneExpressionFile, sep='\t')
    eMat = pd.read_csv(expressionMatrixFile, sep='\t')

    # FILTERING THE VARIABLES WITH TOO MANY NaNs
    gExp['nanCount'] = gExp.apply(lambda row: row.isnull().sum(), axis=1)
    gExpShort = gExp[gExp['nanCount'] < 106]
    gExpID = gExpShort['ID']
    gExpShort = gExpShort.drop(['nanCount', 'ID'], axis=1)
    gExpCols = gExpShort.columns

    # IMPUTE THE MISSING VALUES
    MyImputer = SimpleImputer()
    gExpImputed = pd.DataFrame(MyImputer.fit_transform(gExpShort), columns=gExpCols, index=gExpID)
    return eMat, gExpImputed

def grid_generator():
    params = []
    batch_size = [16, 32, 64]
    layer_size = [90, 80, 70, 60]
    dropout_size = [.1, .3, .5, .7]
    layer_number = [10, 8, 6, 4, 2]
    epoch = [1, 5, 10]
    for bs in batch_size:
        for ls in layer_size:
            for dl in layer_number:
                for ds in dropout_size:
                    params.append({'batch_size': bs,
                                   'layer_size': ls,
                                   'layer_number': dl,
                                   'dropout_size': ds
                                  })
    return params


def StackedAutoencoder(gene_exp_dim, layer_size, dropout_size, encoding_layer):
    input_exp = Input(shape=(gene_exp_dim,))
    # making the encoding Layers
    encoded = Dense(layer_size, activation='relu')(input_exp)
    encoded.add(Dropout(dropout_size))(encoded)
    last_layer = 0
    for enc_layer in range(1, encoding_layer):
        encoded = Dense((layer_size - enc_layer * 2), activation='sigmoid')(encoded)
        encoded = Dropout(dropout_size)(encoded)
        last_layer = layer_size - enc_layer * 2

    ## making the decoding Layers
    decoded = Dense(last_layer, activation='sigmoid')(encoded)
    for dec_layer in range(1, (encoding_layer - 1)):
        decoded = Dense((dec_layer + dec_layer * 2), activation='sigmoid')(decoded)
        decoded = Dropout(dropout_size)(decoded)
    decoded = Dense(shape=(gene_exp_dim,), activation='sigmoid')(encoded)

    return decoded


if __name__ == "__main__":
    expressionMatrixFile = 'data/matrix_genotypes.txt'
    geneExpressionFile = 'data/expression_matrix.txt'
    expressionMatrix, geneExpressionMatrix = reader(expressionMatrixFile, geneExpressionFile)
    gridParameters = grid_generator()
    for hyperParameter in gridParameters:
        decoded = StackedAutoencoder(
            gene_exp_dim=112,
            layer_size=hyperParameter['layer_size'],
            dropout_size=hyperParameter['dropout_size'],
            encoding_layer=hyperParameter['encoding_layer']
        )

        autoEncoder = Model(input_exp, decoded)
        autoEncoder.compile(optimizer='adadelta', loss='mean_squared_error', metrics=['accuracy', 'loss'])
        autoEncoder.fit(
            x_train, x_train,
            epochs=50,
            batch_size=hyperParameter['batch_size'],
            shuffle=True,
            validation_data=(x_test, x_test)
        )
    print('start here')
