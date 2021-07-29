def extract_text_features(tokens):
        """
        Args:
            tokens: A list of tokens, where each token consists of a word,
                optionally followed by u"￨"-delimited features.
        Returns:
            A sequence of words, a sequence of features, and num of features.
        """
        if not tokens:
            return [], [], -1

        specials = ['pad', 'UNK_WORD', 'BOS_WORD', 'DOS_WORD']
        words = []
        features = []
        n_feats = None
        for token in tokens:
            split_token = token.split(u"￨")
            assert all([special != split_token[0] for special in specials]), \
                "Dataset cannot contain Special Tokens"
            #print(split_token)  ['STRING']
            if split_token[0]:
                words += [split_token[0]]
                features += [split_token[1:]]

                if n_feats is None:
                    n_feats = len(split_token)
                else:
                    assert len(split_token) == n_feats, \
                        "all words must have the same number of features"
        features = list(zip(*features))
        print(words)   #['protected', 'boolean', '[', ']', 'can', 'Handle', 'Missing', '(',....]
        print(features) #[]
        print(n_feats)  #1
        return tuple(words), features, n_feats - 1



s='protected boolean [ ] can Handle Missing ( boolean nominal Predictor , boolean numeric Predictor , boolean string Predictor , boolean date Predictor , boolean relational Predictor , boolean multi Instance , int class Type , boolean predictor Missing , boolean class Missing ) { print ( STRING ) ; print Attribute Summary ( nominal Predictor , numeric Predictor , string Predictor , date Predictor , relational Predictor , multi Instance , class Type ) ; print ( STRING ) ; int num Train = get Num Instances ( ) , num Classes =  NUM , missing Level =  NUM ; boolean [ ] result = new boolean [  NUM ] ; Instances insts = null ; Kernel kernel = null ; try { insts = make Test Dataset (  NUM , num Train , nominal Predictor ? get Num Nominal ( ) :  NUM , numeric Predictor ? get Num Numeric ( ) :  NUM , string Predictor ? get Num String ( ) :  NUM , date Predictor ? get Num Date ( ) :  NUM , relational Predictor ? get Num Relational ( ) :  NUM , num Classes , class Type , multi Instance ) ; if ( missing Level >  NUM ) { add Missing ( insts , missing Level , predictor Missing , class Missing ) ; } kernel = Kernel . make Copies ( get Kernel ( ) ,  NUM ) [  NUM ] ; } catch ( Exception ex ) { throw new Error ( STRING + ex . get Message ( ) ) ; } try { Instances train Copy = new Instances ( insts ) ; kernel . build Kernel ( train Copy ) ; compare Datasets ( insts , train Copy ) ; println ( STRING ) ; result [  NUM ] =  BOOL ; } catch ( Exception ex ) { println ( STRING ) ; result [  NUM ] =  BOOL ; if ( m Debug ) { println ( STRING ) ; print ( STRING ) ; println ( STRING + ex . get Message ( ) + STRING ) ; println ( STRING ) ; println ( STRING + insts . to String ( ) + STRING ) ; } } return result ; }'

extract_text_features(s.split())