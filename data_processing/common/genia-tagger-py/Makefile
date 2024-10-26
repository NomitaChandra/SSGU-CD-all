CXX=clang++
CC=clang
CXXFLAGS += -isysroot $(xcrun --show-sdk-path)


# genia_tagger:
# 	wget http://www.nactem.ac.uk/GENIA/tagger/geniatagger-3.0.2.tar.gz
# 	tar -xvzf geniatagger-3.0.2.tar.gz
# 	rm -rf geniatagger-3.0.2.tar.gz/
# 	$(MAKE) -C geniatagger-3.0.2

genia_tagger:
	tar -xvzf /Users/kavithakamarthy/Downloads/SSGU-CD/data_processing/common/genia-tagger-py/geniatagger-3.0.2.tar.gz
	$(MAKE) -C geniatagger-3.0.2

clean:
	rm -f *.o geniatagger

