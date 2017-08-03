---
layout: page
title: ''
---
# Tutorials

This is a living, breathing document. Feel free to add your work below.

{% assign default_paths = site.pages | map: "path" %}
{% assign page_paths = site.header_pages | default: default_paths %}

<ul class="index">
  {% for path in default_paths %}
    {% assign my_page = site.pages | where: "path", path | first %}
    {% assign title = my_page.title | trim %}
    {% if title %}
    <li>
      <a href="{{ my_page.url | relative_url }}">
        {{my_page.title | escape }}
      </a>
    </li>
    {% endif %}
  {% endfor %}
</ul>
