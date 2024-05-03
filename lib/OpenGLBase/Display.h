/*
* This portion of code originally made by BennyQBD
* https://github.com/BennyQBD/ModernOpenGLTutorial/blob/master/
*/

#include <iostream>
#include <SDL2/SDL.h>
#include <GL/glew.h>

// #include "Shader.h"

class Display
{
public:
    Display(int width, int height, const std::string& title)
    {
        SDL_Init(SDL_INIT_EVERYTHING);
        SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
        SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE,32);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,16);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER,1);

        window = SDL_CreateWindow(
            title.c_str(),
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            width,
            height,
            SDL_WINDOW_OPENGL);
        GLContext = SDL_GL_CreateContext(window);

#if defined DEBUG_MODE || defined TEST_MODE
        std::cout << "Window " << title << " created successfully." << std::endl;
#endif

        GLenum err = glewInit();

        if (err != GLEW_OK)
        {
            std::cerr << "Failed to initialize GLEW." << std::endl;
        }
#if defined DEBUG_MODE
        else
        {
            std::cout << "Successfully initialized GLEW." << std::endl;
        }
#endif
    	glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);
    }

    virtual ~Display()
    {
        SDL_GL_DeleteContext(GLContext);
        SDL_DestroyWindow(window);
        SDL_Quit();

#if defined DEBUG_MODE || defined TEST_MODE
        std::cout << "Window destructed successfully." << std::endl;
#endif
    }

    void Update() 
    { 
        SDL_GL_SwapWindow(window); 

        SDL_Event e;
        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT)
                isClosed = true;
        }
    }

    void Clear(float r, float g, float b, float alpha = 1.f)
    {
        glClearColor(r, g, b, alpha);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }

    bool inline getIsClosed() { return isClosed; };

protected:
private:
    Display(const Display& other) {}
    Display& operator=(const Display& other) { return *this; }

    SDL_Window* window;
    SDL_GLContext GLContext;

    bool isClosed = false;
};