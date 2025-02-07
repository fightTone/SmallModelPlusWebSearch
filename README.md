## REQUIREMENTS

1. ollama
2. python
3. npm

## Run the FastAPI
1. ```cd api```
2. ```pip install -r requirements.txt```
3. ```python api.py```


## Run the interface

1. Install tailwindcss and its peer dependencies properly:

    ```npm install -D tailwindcss@3.4.1 postcss@8.4.35 autoprefixer@10.4.17```

2. Now try initializing Tailwind again (note the package name is 'tailwindcss', not 'tailwind'):

    ```npx tailwindcss@3.4.1 init -p```

3. If that doesn't work, we can manually create both configuration files.

In tailwind.config.js:

```
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```
In postcss.config.js:
```
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

Install other needed packages:

```npm install lucide-react @headlessui/react```

Finally:

```npm start```

### sample pics
![Screenshot](https://github.com/fightTone/SmallModelPlusWebSearch/blob/main/sample_pics/api.png)
![Screenshot](https://github.com/fightTone/SmallModelPlusWebSearch/blob/main/sample_pics/interface.png)
