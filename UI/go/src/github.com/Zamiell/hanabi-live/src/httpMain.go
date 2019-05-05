package main

import (
	"github.com/gin-gonic/gin"
)

func httpMain(c *gin.Context) {
	w := c.Writer
	data := TemplateData{
		Title: "Dev",
	}
	httpServeTemplate(w, data, "main")
}
