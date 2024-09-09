from math import ceil
from random import choices, choice, randint
import copy
from extract_features_html import isInternal, getObjects
from tld import get_tld
import base64


class Manipulation:
    def __call__(self, html_obj, url):
        raise NotImplementedError("The manipulation functionality must me defined by overriding __call__()")


class InjectIntElem(Manipulation):
    def __call__(self, html_obj, url, obj_type='a', to_footer=False, num_objs=10, use_hidden=True):
        soup_obj = copy.copy(html_obj)

        try:
            assert num_objs is None or (isinstance(num_objs, int) and num_objs > 0)
        except AssertionError:
            raise Exception("num_obj: {}".format(num_objs))
        assert obj_type in {'a', 'link', 'img', 'script', 'meta'}

        num_objs = num_objs if num_objs else 10  # randint(10, 80)

        if to_footer:
            if soup_obj.footer is None:
                if soup_obj.html is not None and soup_obj.html.body is not None:
                    soup_obj.html.body.append(soup_obj.new_tag('footer', attrs={'hidden': 'hidden'}))
                else:
                    raise Exception("No body or html found to append footer")
            parent_obj = soup_obj.html.body.footer if soup_obj.html and soup_obj.html.body else None
        else:
            if use_hidden:
                parent_obj = soup_obj.html.body if soup_obj.html and soup_obj.html.body else None
            else:
                new_tag_noscript = soup_obj.new_tag('noscript')
                if soup_obj.html and soup_obj.html.body:
                    soup_obj.html.body.append(new_tag_noscript)
                else:
                    raise Exception("No body or html found to append noscript")
                parent_obj = new_tag_noscript

        if parent_obj is None:
            raise Exception("parent_obj is None. Could not find a valid parent element to append.")

        link_str = ['#!', '#null', '#injected']
        links = choices(link_str, k=num_objs)

        for link in links:
            if obj_type == 'a':
                attrs = {'href': link}
            elif obj_type == 'link':
                attrs = {'href': link, 'rel': 'help'}
            elif obj_type == 'img':
                attrs = {'src': link, 'alt': 'example'}
            elif obj_type == 'script':
                attrs = {'src': link}
            elif obj_type == 'meta':
                attrs = {'http-equiv': 'refresh', 'content': '3;URL="{}"'.format(link)}
            if use_hidden:
                attrs['hidden'] = 'hidden'
            injected_tag = soup_obj.new_tag(obj_type, attrs=attrs)
            if obj_type == 'a':
                injected_tag.string = "Link"
            parent_obj.append(injected_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'IIE'


class InjectIntElemFoot(InjectIntElem):
    def __call__(self, html_obj, url):
        return super().__call__(html_obj, url, obj_type='a', to_footer=True)

    def __str__(self) -> str:
        return 'IIF'


class InjectIntLinkElem(InjectIntElem):
    def __call__(self, html_obj, url):
        return super().__call__(html_obj, url, obj_type='link')

    def __str__(self) -> str:
        return 'IIL'


class InjectExtElem(Manipulation):
    def __call__(self, html_obj, url, obj_type='link', num_objs=10, to_footer=False, use_hidden=True):
        soup_obj = copy.copy(html_obj)

        assert num_objs is None or (isinstance(num_objs, int) and num_objs > 0)
        assert obj_type in {'a', 'link', 'img', 'script', 'meta'}

        num_objs = num_objs if num_objs else 10  # randint(10, 80)

        if to_footer:
            if soup_obj.footer is None:
                soup_obj.html.body.append(soup_obj.new_tag('footer', attrs={'hidden': 'hidden'}))
            parent_obj = soup_obj.html.body.footer
        else:
            if use_hidden:
                parent_obj = soup_obj.html.body
            else:
                new_tag_noscript = soup_obj.new_tag('noscript')
                soup_obj.html.body.append(new_tag_noscript)
                parent_obj = new_tag_noscript

        links_ext = [
            'https://www.google.com',
            'https://www.youtube.com',
            'https://www.facebook.com',
            'https://www.wikipedia.org',
            'https://www.amazon.com',
            'https://www.baidu.com',
            'https://www.openai.com',
            'https://www.microsoft.com',
            'https://www.ebay.com',
            'https://www.quora.com',
            'https://www.zoom.us',
            'https://www.discord.com',
            'https://www.msn.com',
            'https://www.outlook.com',
            'https://www.duckduckgo.com',
            'https://www.linkedin.com',
            'https://www.netflix.com',
            'https://www.eurecom.fr',
        ]
        links = choices([link for link in links_ext if not isInternal(link, url)], k=num_objs)

        for link in links:
            if obj_type == 'a':
                attrs = {'href': link}
            elif obj_type == 'link':
                attrs = {'href': link, 'rel': 'help'}
            elif obj_type == 'img':
                attrs = {'src': link, 'alt': 'example'}
            if use_hidden:
                attrs['hidden'] = 'hidden'
            injected_tag = soup_obj.new_tag(obj_type, attrs=attrs)
            if obj_type == 'a':
                injected_tag.string = "Link"
            parent_obj.append(injected_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'IEE'


class InjectExtElemFoot(InjectExtElem):
    def __call__(self, html_obj, url):
        return super().__call__(html_obj, url, obj_type='link')

    def __str__(self) -> str:
        return 'IEF'


class UpdateForm(Manipulation):
    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)
        forms = soup_obj.findAll("form")
        checked_actions = {"", "#", "#nothing", "#doesnotexist", "#null", "#void", "#whatever", "#content",
                           "javascript::void(0)", "javascript::void(0);", "javascript::;", "javascript"}

        for form in forms:
            if 'action' in form.attrs:
                if form['action'] in checked_actions:
                    form['action'] = choice(['_none.php', '#none', '#!'])

        return soup_obj

    def __str__(self) -> str:
        return 'UPF'


class ObfuscateExtLinks(Manipulation):
    def __init__(self) -> None:
        self._obj_id = 0

    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)

        html_tags = getObjects(soup_obj)
        forms = soup_obj.findAll("form")
        html_tags.extend(forms)
        if len(html_tags) == 0:
            return soup_obj

        ext_link_replace = []
        for obj in html_tags:
            if 'src' in obj.attrs:
                obj_link = obj['src']
                obj_attr = 'src'
            elif 'href' in obj.attrs:
                obj_link = obj['href']
                obj_attr = 'href'
            elif 'action' in obj.attrs:
                obj_link = obj['action']
                obj_attr = 'action'
            else:
                continue

            if 'id' in obj.attrs:
                obj_id = obj['id']
            else:
                obj['id'] = '_ext{}'.format(self._obj_id)
                self._obj_id += 1
                obj_id = obj['id']

            if not isInternal(obj_link, url):
                obj[obj_attr] = "#!"
                # del obj[obj_attr]
                ext_link_replace.append((obj_id, obj_attr, obj_link))

        if len(ext_link_replace) == 0:
            return soup_obj

        script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
        code = "window.onload = function () {\n"
        for value in ext_link_replace:
            obj_id, obj_attr, obj_link = value
            code += "    document.getElementById('{id}').setAttribute('{attr}', '{link}');\n".format(id=obj_id, attr=obj_attr, link=obj_link)
        code += "  }"
        script_tag.string = code
        soup_obj.html.head.append(script_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'OEL'


class UpdateHiddenDivs(Manipulation):
    def __init__(self) -> None:
        self._obj_id = 0

    def __call__(self, html_obj, url):

        soup_obj = copy.copy(html_obj)
        divs = soup_obj.findAll("div")

        for div in divs:
            if 'style' in div.attrs:
                hidden_mode = None
                for val in div['style'].split(';'):
                    if val in ['visibility:hidden', 'visibility: hidden']:
                        hidden_mode = 'visibility_hidden'
                        break
                    elif val in ['display:none', 'display: none']:
                        hidden_mode = 'display_none'
                        break

                if hidden_mode is not None:
                    new_style_values = ';'.join([val for val in div['style'].split(';')
                                                 if val not in ['visibility:hidden', 'visibility: hidden', 'display:none', 'display: none']])
                    div['style'] = new_style_values

                    if hidden_mode == 'display_none':
                        div['hidden'] = ""
                    else:
                        if 'id' in div.attrs:
                            obj_id = div['id']
                        else:
                            div['id'] = '_div{}'.format(self._obj_id)
                            self._obj_id += 1
                            obj_id = div['id']

                        style_tag = soup_obj.new_tag('style')
                        style_tag.string = "#{} {{ visibility: hidden; }}".format(obj_id)
                        soup_obj.html.head.append(style_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'UHD'


class UpdateHiddenButtons(Manipulation):
    def __init__(self) -> None:
        self._obj_id = 0

    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)
        buttons = soup_obj.findAll("button")

        for button in buttons:
            if 'disabled' in button.attrs:
                del button['disabled']

                if 'id' in button.attrs:
                    obj_id = button['id']
                else:
                    button['id'] = '_button{}'.format(self._obj_id)
                    self._obj_id += 1
                    obj_id = button['id']

                script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
                script_tag.string = "document.getElementById('{id}').setAttribute('disabled', '');".format(id=obj_id)
                button.insert_after(script_tag)
        return soup_obj

    def __str__(self) -> str:
        return 'UHB'


class UpdateHiddenInputs(Manipulation):
    def __init__(self) -> None:
        self._obj_id = 0

    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)
        inputs = soup_obj.findAll("input")

        for inp in inputs:
            if inp.attrs.get('type') == 'hidden':
                inp['type'] = 'text'
                inp['hidden'] = ""
            elif 'disabled' in inp.attrs:
                del inp['disabled']

                if 'id' in inp.attrs:
                    obj_id = inp['id']
                else:
                    inp['id'] = '_input{}'.format(self._obj_id)
                    self._obj_id += 1
                    obj_id = inp['id']

                script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
                script_tag.string = "document.getElementById('{id}').setAttribute('disabled', '');".format(id=obj_id)
                inp.insert_after(script_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'UHI'


class UpdateIntAnchors(Manipulation):
    def __init__(self) -> None:
        self._obj_id = 0

    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)
        anchors = soup_obj.findAll("a")
        for a in anchors:
            if 'href' in a.attrs:
                anchor_location = a['href']
                # if isInternal(anchor_location, url):
                if anchor_location in ["#", "#content", "#skip", "JavaScript ::void(0)", "javascript::void(0)"]:
                    a['href'] = "#!"

        return soup_obj

    def __str__(self) -> str:
        return 'UIA'


class InjectFakeCopyright(Manipulation):
    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)
        try:
            res = get_tld(url, as_object=True)
        except Exception:
            return soup_obj
        domain = res.domain
        copyright_symbol = u'\N{COPYRIGHT SIGN}'.encode('utf-8')
        copyright_symbol = copyright_symbol.decode('utf-8')

        new_p_tag = soup_obj.new_tag('p', attrs={"hidden": ""})
        new_p_tag.string = "{} Copyright {}".format(copyright_symbol, domain)
        soup_obj.html.body.append(new_p_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'IFC'


class UpdateTitle(Manipulation):
    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)

        try:
            domain_brand = get_tld(url, as_object=True).domain
        except Exception:
            return html_obj

        title = html_obj.find('title')

        if title and title.string:
            original_title = title.string.strip()
            soup_obj.title.string = domain_brand.strip()

            script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
            script_tag.string = "document.title = \"{}\";".format(original_title)
            soup_obj.html.head.append(script_tag)
        else:
            new_title = soup_obj.new_tag('title', attrs={'id': 'title'})
            new_title.string = domain_brand.strip()

            script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
            script_tag.string = "document.getElementById('title').remove();"
            soup_obj.html.head.append(new_title)
            soup_obj.html.head.append(script_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'UPT'


class UpdateIFrames(Manipulation):
    def __init__(self) -> None:
        self._obj_id = 0

    # def update_iframes(html_obj, url):
    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)

        iframes = soup_obj.find_all('iframe')
        for iframe in iframes:
            hidden_mode = None
            if 'style' in iframe.attrs:
                for val in iframe['style'].split(';'):
                    if val in ['visibility:hidden', 'visibility: hidden']:
                        hidden_mode = 'visibility_hidden'
                        break
                    elif val in ['display:none', 'display: none']:
                        hidden_mode = 'display_none'
                        break
                    elif val in ['border: 0', 'border:0']:
                        hidden_mode = 'border_0'
                        break

                if hidden_mode is not None:
                    new_style_values = ';'.join([val for val in iframe['style'].split(';')
                                                 if val not in ['visibility:hidden', 'visibility: hidden', 'display:none', 'display: none']])
                    iframe['style'] = new_style_values

                    if hidden_mode == 'display_none':
                        iframe['hidden'] = ""
                    else:
                        if hidden_mode == 'visibility_hidden':
                            style = "visibility: hidden;"
                        elif hidden_mode == 'border_0':
                            style = "border: 0;"

                        if 'id' in iframe.attrs:
                            obj_id = iframe['id']
                        else:
                            iframe['id'] = '_iframe{}'.format(self._obj_id)
                            self._obj_id += 1
                            obj_id = iframe['id']

                        style_tag = soup_obj.new_tag('style')
                        style_tag.string = "#{id} {{ {style} }}".format(id=obj_id, style=style)
                        soup_obj.html.head.append(style_tag)

            if 'frameborder' in iframe.attrs and iframe['frameborder'] == '0':
                if 'id' in iframe.attrs:
                    obj_id = iframe['id']
                else:
                    iframe['id'] = '_iframe{}'.format(self._obj_id)
                    self._obj_id += 1
                    obj_id = iframe['id']

                    script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
                    script_tag.string = "document.getElementById('{id}').setAttribute('frameborder', '0');".format(id=obj_id)
                    soup_obj.head.append(script_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'UIF'


class InjectFakeFavicon(Manipulation):
    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)
        # if no favicon is in the original web page add inject a fake one
        if html_obj.find(rel="shortcut icon") is None and html_obj.find(rel="icon") is None:
            fake_favicon = soup_obj.new_tag('link', attrs={'rel': 'shortcut icon', 'href': '#none', 'hidden': ''})
            soup_obj.html.head.append(fake_favicon)

        return soup_obj

    def __str__(self) -> str:
        return 'IFF'


class ObfuscateJS(Manipulation):
    def __init__(self) -> None:
        self._obj_id = 0

    def __call__(self, html_obj, url, patterns=[]):
        soup_obj = copy.copy(html_obj)
        assert isinstance(patterns, list)

        # Default patterns to look for in JavaScript code
        if not patterns:
            patterns = ['window.open(', 'prompt(', 'preventDefault()', 'window.status']

        # Process script tags with inline JavaScript
        script_tags = soup_obj.find_all('script', string=True)
        for script in script_tags:
            original_script = script.string

            # Reverse the original script content to obfuscate it
            obfuscated_script = ''.join(reversed(original_script))

            # Replace the original script content with the obfuscated version
            script.string.replace_with(obfuscated_script)

        # Process script tags and elements with onload attributes
        scripts = soup_obj.find_all('script')
        elems_onload = soup_obj.select('[onload]')
        target_elems = scripts + elems_onload
        
        for elem in target_elems:
            for pattern in patterns:
                code = elem.string if elem.name == "script" else elem.get('onload', '')
                if code and pattern in code:
                    # 1) extract JS code and encode into Base64
                    script_code = code.encode('utf-8')
                    encoded_script = base64.b64encode(script_code).decode('utf-8')
                    # 2) Create a new script that will include the obfuscated JS code
                    parent_elem = 'head' if elem.parent.name == 'head' else 'body'
                    obf_script_code = """
                    let script = document.createElement("script");
                    \t script.innerHTML = atob("{encoded_script}");
                    \t document.{parent_type}.appendChild(script);
                    """.format(parent_type=parent_elem, encoded_script=encoded_script)
                    # 3) Overwrite the original code with obfuscated code
                    if elem.name == "script":
                        elem.string.replace_with(obf_script_code)
                    else:
                        elem['onload'] = obf_script_code

        # Process elements with oncontextmenu attributes
        target_elems = soup_obj.select('[oncontextmenu]')
        for elem in target_elems:
            # remove oncontextmenu and create a new script to re-add it when the webpage is loaded
            code = elem['oncontextmenu']
            del elem['oncontextmenu']
            script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
            if 'id' in elem.attrs:
                elem_id = elem['id']
            else:
                elem['id'] = '_obj{}'.format(self._obj_id)
                self._obj_id += 1
                elem_id = elem['id']
            script_tag.string = 'document.getElementById("{id}").setAttribute(atob("b25jb250ZXh0bWVudQ=="), "{value}");'.format(id=elem_id, value=code)

            elem.insert_after(script_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'OJS'


# Manipulations related to the MLSEC challenge
class InjectHiddenForms(Manipulation):
    def __call__(self, html_obj, url, num_objs=10):
        soup_obj = copy.copy(html_obj)
        num_objs = 20  # randint(5, 20)
        use_hidden = True  # choice([True, False])

        has_body = (soup_obj.html.body is not None)
        main_tag = soup_obj.html.body if has_body else soup_obj.html

        if not use_hidden:
            new_tag_noscript = soup_obj.new_tag('noscript')
            main_tag.append(new_tag_noscript)
            parent_obj = new_tag_noscript
        else:
            parent_obj = main_tag

        for _ in range(num_objs):
            form_action = choice(['_none.php', '#none', '#!'])
            attrs = {'action': form_action}
            if use_hidden:
                attrs = {'hidden': None}
            new_form = soup_obj.new_tag('form', attrs=attrs)
            new_form.append(soup_obj.new_tag('input', attrs={'hidden': None, 'type': 'text'}))
            parent_obj.append(new_form)

        return soup_obj

    def __str__(self) -> str:
        return 'IHF'


class UpdateInputPasswd(Manipulation):
    def __call__(self, html_obj, url):
        soup_obj = copy.copy(html_obj)
        inputs = soup_obj.findAll("input")

        for inp in inputs:
            if 'type' in inp and inp['type'] == 'password':
                if ('value' not in inp.attrs) or (inp.attrs.get('value', None) == ""):
                    inp['type'] = 'text'

        return soup_obj

    def __str__(self) -> str:
        return 'UIP'


class InjectJS(Manipulation):
    def __call__(self, html_obj, url, js_code):
        soup_obj = copy.copy(html_obj)
        num_objs = randint(5, 20)
        use_noscript = choice([True, False])
        to_head = choice([True, False])
        assert js_code is None or isinstance(js_code, str)

        has_body = (soup_obj.html.body is not None)
        has_head = (soup_obj.html.head is not None)
        if to_head and has_head:
            main_tag = soup_obj.html.head
        elif has_body:
            main_tag = soup_obj.html.body
        else:
            main_tag = soup_obj.html

        if use_noscript:
            new_tag_noscript = soup_obj.new_tag('noscript')
            main_tag.append(new_tag_noscript)
            parent_obj = new_tag_noscript
        else:
            parent_obj = main_tag

        random_js_code = [
            "var temp = (new Date()).getTime();",
            "function foo() {var baz = ""; return 1;}",
            ""
        ]
        for _ in range(num_objs):
            script_tag = soup_obj.new_tag('script', attrs={'type': 'text/javascript'})
            injected_code = js_code if js_code is not None else choice(random_js_code)
            script_tag.string = injected_code
            parent_obj.append(script_tag)

        return soup_obj

    def __str__(self) -> str:
        return 'IJS'


class InjectIntElemThreshold(InjectIntElem):
    def __call__(self, html_obj, url, obj_type='a', susp_thr=0.15):
        html_tags = getObjects(html_obj)
        if len(html_tags) == 0:
            return html_obj
        else:
            num_ext_links = 0
            num_tot_links = 0
            for obj in html_tags:
                if 'src' in obj.attrs:
                    object_location = obj['src']
                elif 'href' in obj.attrs:
                    object_location = obj['href']
                else:
                    continue

                num_tot_links += 1

                if not isInternal(object_location, url):
                    num_ext_links += 1

            if num_tot_links == 0 or num_ext_links == 0:
                return html_obj

            num_int_links = ceil(((1 - susp_thr) / susp_thr) * num_ext_links)
            return super(InjectIntElemThreshold, self).__call__(html_obj, url, obj_type=obj_type, num_objs=num_int_links)

    def __str__(self) -> str:
        return 'IIO'

