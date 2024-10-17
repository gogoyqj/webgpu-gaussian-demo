const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  lintOnSave: false,
  devServer: {
  },
  configureWebpack: {
    module: {
      rules: [
        {
          test: /\.wgsl$/,
          use: 'raw-loader',
        }
      ]
    }
  },
})
