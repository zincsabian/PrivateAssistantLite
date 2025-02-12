from pipeline import QAPipeline

if __name__ == "__main__":
    test_text = """
<think>
嗯，我现在要回答用户的问题：“什么是SQL注入攻击”。首先，我需要了解SQL注入是什么，它如何工作，以及为什么会有这样的漏洞存在。

我记得SQL是 Structured Query Language，用于访问和管理数据库的编程语言。注入攻击可能涉及到将恶意代码插入到数据库查询中，让攻击者执行任意操作。

接下来，用户想知道注入攻击是如何工作的。可能涉及利用不完整的数据库权限或结构，比如 injection points 未被正确验证，或者注入的SQL语句不完整，从而允许攻击者执行SELECT、INSERT等命令。

还要考虑注入攻击的目标是什么，可能是窃取数据、删除数据或修改数据。攻击者通常会使用HTTP请求中的参数来隐藏恶意代码，可能通过CSRF保护机制绕过防 SQL 注入。

我还需要了解如何防御SQL注入攻击。常见的防护措施包括验证输入字段的完整性，使用编码技术如SQL Sanitization，启用数据库的SQL过滤功能，以及在数据库层面应用加密和访问控制。

另外，检查PHP和框架中的漏洞也很重要，因为它们是常见的注入攻击目标。同时，定期进行安全审计和代码审查也能帮助发现潜在的风险。

最后，总结一下SQL注入攻击的类型、工作原理、影响以及防御方法，这样用户就能全面理解这个问题，并采取相应的措施来防范。
</think>

为了更好地理解 SQL 注入攻击，请考虑以下问题：

1. **SQL 注入攻击是如何工作的？**
2. **注入攻击的目标是什么？**
3. **常见的 SQL 注入攻击有哪些类型？**
4. **如何识别一个可能的 SQL 注入漏洞？**
5. **防御 SQL 注入攻击需要哪些安全措施？**

这些问题将帮助你深入了解 SQL 注入攻击的本质、影响以及防范方法。
"""
    
    pipeline = QAPipeline(log_level="INFO")
    queries = pipeline.extract_search_queries(test_text)
    print("\nExtracted question chunks:")
    for i, chunk in enumerate(queries, 1):
        print(f"\nChunk {i}:")
        print(chunk)