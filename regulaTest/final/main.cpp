#include <iostream>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <face_detector.h>
#include <string>
namespace bf = boost::filesystem;

void proccesFolder(const bf::path & p,const int & workers)
{
    bf::directory_iterator endItr;
    for (bf::directory_iterator dirItr(p); dirItr != endItr; ++dirItr)
    {
        if (bf::is_directory(dirItr->status()))
            proccesFolder(dirItr->path(), workers);
        else {
            proccesImg(dirItr->path().filename().string(),p.string()+"/");
        }

    }
}

int main(int argc, char ** argv)
{
    std::string pathFolder = argv[1];
    int workers = 4;
    if (argc == 3)
        workers = std::stoi(argv[2]);
    bf::path p(pathFolder);
    proccesFolder(p,workers);
    return 0;
}
