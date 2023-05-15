#!/usr/bin/perl -w
#
# A simple content-word selector.
#
# chasen < euc-jp-text | cstemmer.pl
#
# NOTE:
# 
# ChaSen version 2.2.1 is assumed.
# This file should be saved in EUC-JP.
#

@words = ();
while(<>){
    if(/EOS/){
	print "@words\n";
	@words = ();
    }else{
	my ($word,$pron,$lemma,$pos) = split;
	push(@words, $lemma) if isContentWord($lemma,$pos);
    }
}

sub isContentWord {		# refine ����ɬ�פ����롥
    my $lemma = shift;
    my $pos = shift;
    
    # ascii
    if($lemma =~ /^[\000-\177]+$/){ 
	return 0 if $lemma=~/^[^a-zA-Z]+$/;
    }
    
    # not ascii
    return 1 if $lemma=~/^([\245].)+$/;	# �Ҳ�̾
    return 0 if $lemma =~ /^((\244.)+)$/; # hiragana sequence
    return 1 if $pos =~ /̤�θ�/;
    
    return 1 if $pos =~ /^����-����ե��٥å�/;
    return 1 if $pos =~ /^̾��/ && $pos !~/̾��-(��|��̾��|��Ω|�ü�|����|��³|ư��)/;
    return 1 if $pos =~ /^(��Ƭ|̾��-����)/;
    return 0;
}
