# this script needs spacy 1.x
import click as click
import spacy


@click.command()
@click.argument('embedding_model_txt', default='/home/lukasz/data/glove.6B.50d.txt', type=click.Path(exists=True))
@click.argument('embedding_model_bin', default='/home/lukasz/data/glove.6B.50d.bin', type=click.Path(exists=False))
def parse_txt_to_bin(embedding_model_txt, embedding_model_bin):
    model = spacy.load('en', vectors=False)
    with open(embedding_model_txt, 'r', encoding='utf-8') as f:
        model.vocab.load_vectors(f)
        click.echo(f'Vectors {embedding_model_txt} loaded.')
    model.to_disk(f'{embedding_model_bin}')
    click.echo(f'Vectors {embedding_model_bin} saved.')


if __name__ == '__main__':
    parse_txt_to_bin()
