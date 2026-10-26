Reposta para as perguntas 
Alunos, Josué, Alan Pereira e Raphael

1. modelo é deep learning, ele faz uso de tensor flow/keras e manipula camadas de data augmentation e tensores, o código tem uma arquitetura de várias camadas, o que é tipico de deep learning com entrada e saída. 

2. O modelo usa as bordas circulares dos dedos e a posição deles para encontrar os formatos, percebemos que é muito mais fácil ele reconhecer os gestos de pedra papel e tesoura do que os sináis de libras corretamente.

3. Usamos duas abordagens na coleta de dados. A primeira foi a mudança de fundo, gravando os sinais primeiro em um fundo mais claro e depois em um mais escuro, isso ajudou porque o modelo passou a olhar para a mão e não para o fundo, mas dificultou porque no fundo escuro o contraste com a mão é menor e as bordas dos dedos ficam menos visíveis, por isso a acurácia na validação ficou mais baixa que no treino. A segunda foi treinar usando apenas uma das mãos, isso ajudou porque cada letra fica sempre na mesma posição e o modelo consegue separar melhor sinais parecidos como V e W, mas dificultou porque o modelo não reconhece bem quando o sinal é feito com a outra mão ou por outra pessoa.

4. Durante o treinamento não foi possivel criar cenários com pessoas ao fundo, por esse motivo o resultado final ficou projudicado, mas podemos citar que em alguns casos ocorre a não indentificação do sinal por conta de não focar somente em um contorno que é o necessário para identificação de gestos.